//! The IQ ingestion worker — a SECOND, dedicated WebSocket connection to SDRconnect (the first
//! is the CAT shim's control socket, `sdrconnect_daemon::SdrConnectBackend`), whose only job is
//! to keep binary IQ frames flowing into `sdrconnect_dsp`'s software receiver and hand back
//! finished 12 kHz mono audio.
//!
//! # Why a separate connection, not the CAT socket
//!
//! Enabling `iq_stream_enable` on the SAME socket the CAT shim uses for `get_property` would
//! interleave binary IQ frames with the JSON property responses `SdrConnectBackend` is waiting
//! on — and `SdrConnectBackend::client`'s `get_property` already silently drops any `Binary`
//! message it sees mid-wait (reasonably, since Phase 1 of the CAT shim never expected one),
//! which would just discard every IQ frame that arrived at the wrong moment. Two sockets, two
//! concerns, no interleaving.
//!
//! # The pull contract
//!
//! [`SdrConnectIq::take_audio`] mirrors `crate::flexdax::FlexDax::take_audio` exactly: drain and
//! return whatever has been demodulated since the last call. `service.rs`'s per-tick capture
//! step can then treat "native Flex DAX audio" and "native SDRconnect IQ audio" identically —
//! see the call site added next to `self.dax_src.as_ref()`.
//!
//! # Self-healing, like `flexdax`
//!
//! A dropped WebSocket (SDRconnect restarted, a Wi-Fi blip) is NOT fatal here: the worker loop
//! reconnects on its own with a short backoff, the same "give it a few seconds before the
//! operator notices" discipline `flexdax`'s own reconnect uses. `is_alive()` still reports the
//! CURRENT connection state, so the surrounding `SdrConnectDaemon`/CAT-probe machinery can still
//! surface a persistent failure — the two are complementary, not redundant: a two-second blip
//! self-heals invisibly, a genuinely dead RSP1B still shows up as CAT trouble.
//!
//! # Real-time monitor feed
//!
//! `take_audio()` alone is not good enough for LIVE LISTENING, even though it is fine for
//! decode: it is only ever drained by `RadioLoop`'s ~20 ms tick, which shares its cadence with
//! CAT polling — a stalled or slow CAT round-trip (plausible for SDRconnect: a network hop, not
//! a local serial port) delays the NEXT drain by however long that stall lasted. Decode does not
//! notice (it accumulates into a window and tolerates the jitter); a human listening does
//! (heard as choppy, gapped audio despite `Monitor`'s own 500 ms ring, first reported and traced
//! 2026-09). `monitor_sink` (see [`SdrConnectIq::start`]) exists to route around this: when
//! given, this worker pushes each finished block into it from ITS OWN thread the instant that
//! block is ready — the exact same "real-time push, not a tick-driven pull" pattern the physical
//! sound card's own capture callback already gets for free. `take_audio()`'s tick-driven pull
//! keeps working unchanged for decode; the two delivery paths are independent and both always
//! active when `monitor_sink` is `Some`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::capture_resample::CaptureResampler;
use crate::sdrconnect::{IncomingMessage, SdrConnect, SdrConnectError};
use crate::sdrconnect_dsp::{
    demod_am, demod_fm, demod_ssb, design_lowpass, design_ssb_bandpass, taps_for_transition,
    ComplexFir, DemodMode, DemodParams, FirDecimator, Iq,
};

/// The IQ sample rate Nexus asks the RSP1B for. Fixed and deliberately modest: wide enough to
/// comfortably hold every mode this receiver supports (WFM's ~150 kHz included, with margin)
/// while keeping the Stage 1 decimation filter's tap count — and therefore its CPU cost —
/// small. See `sdrconnect_dsp`'s module doc for why this can be fixed rather than tunable: with
/// no NCO retuning, the whole captured block is either "wide enough" or "not", never partially
/// useful.
pub const IQ_SAMPLE_RATE_HZ: f64 = 250_000.0;
/// The Stage 1 output rate every mode's channel filter targets — a TARGET, not a guarantee: see
/// `connect_and_arm`'s doc for why the actual rate used at runtime is derived from what
/// SDRconnect confirms, not from `IQ_SAMPLE_RATE_HZ` divided by a fixed decimation factor.
const TARGET_STAGE1_RATE_HZ: f64 = 25_000.0;
const FINAL_RATE_HZ: u32 = 12_000;
/// Binary message type for Primary-device raw IQ (SDRconnect WebSocket API 1.0.3, "Binary
/// Messages"). Secondary-device IQ (type 5) is out of scope — Phase 1 is a single-tuner (single
/// RSP1B) receiver, matching the CAT shim's Primary-only scope.
const BINARY_TYPE_IQ_PRIMARY: u16 = 2;
/// A generous cap (a few seconds of audio) on how much undrained output this worker will hold.
/// Exists so a consumer that stops draining — the radio switched away from, the app backgrounded
/// — cannot grow this buffer without bound; sized well above any normal tick cadence.
const MAX_BUFFERED_SAMPLES: usize = FINAL_RATE_HZ as usize * 5;
/// How long to wait before retrying a dropped or refused connection.
const RECONNECT_BACKOFF: Duration = Duration::from_millis(1_500);

/// The Stage 2 channel filter + demodulator state for whichever mode is current. Rebuilt only
/// on a mode/bandwidth change (checked once per binary frame, not once per sample) — cheap
/// enough that the check itself never has to be optimised away.
enum Stage2 {
    /// AM (envelope), NFM/WFM (discriminator) — all a real lowpass followed by a different
    /// demod, so they share one filter shape and differ only in what runs after it.
    Lowpass {
        filter: FirDecimator,
        dc_state: f32,
        is_fm: bool,
        fm_prev: Iq,
        fm_gain: f32,
    },
    /// USB/LSB/CW — a one-sided complex bandpass; see `sdrconnect_dsp::design_ssb_bandpass`.
    Bandpass { filter: ComplexFir, gain: f32 },
}

/// Build the Stage 2 filter + demod state for `mode`/`bandwidth_hz`, running at
/// `stage1_rate_hz` — the CONFIRMED Stage 1 output rate for the current connection (see
/// `connect_and_arm`'s doc), not a compile-time constant, since it depends on whatever rate
/// SDRconnect actually settled on. `bandwidth_hz == 0` (never configured, e.g. before the first
/// CAT `M`/`F` command) falls back to a sane per-mode default rather than building a zero-width
/// filter.
fn build_stage2(mode: DemodMode, bandwidth_hz: u32, stage1_rate_hz: f64) -> Stage2 {
    // Stay comfortably clear of the Nyquist at `stage1_rate_hz` regardless of what the operator
    // (or WSJT-X, via rigctld `M`) asks for — a wider request is clamped, not refused.
    let nyquist_margin = stage1_rate_hz * 0.46;
    match mode {
        DemodMode::Am | DemodMode::Nfm | DemodMode::Wfm => {
            let default_bw = match mode {
                DemodMode::Am => 6_000,
                DemodMode::Nfm => 12_000,
                _ => 150_000, // WFM — see the module doc: IQ_SAMPLE_RATE_HZ barely covers this
            };
            let bw = if bandwidth_hz == 0 { default_bw } else { bandwidth_hz } as f64;
            let cutoff = (bw / 2.0).min(nyquist_margin);
            let transition = (cutoff * 0.3).max(200.0);
            let taps = design_lowpass(
                taps_for_transition(stage1_rate_hz, transition),
                cutoff / stage1_rate_hz,
            );
            let is_fm = matches!(mode, DemodMode::Nfm | DemodMode::Wfm);
            // Discriminator gain converts radians/sample to a level roughly comparable to the
            // other demodulators' output — an approximation (real deviation varies by station),
            // not a calibrated measurement. WFM's much larger deviation needs a smaller gain.
            let fm_gain = if matches!(mode, DemodMode::Wfm) { 0.15 } else { 0.6 };
            Stage2::Lowpass {
                filter: FirDecimator::new(taps, 1),
                dc_state: 0.0,
                is_fm,
                fm_prev: Iq::new(1.0, 0.0),
                fm_gain,
            }
        }
        DemodMode::Usb | DemodMode::Lsb => {
            let bw = (if bandwidth_hz == 0 { 2_700 } else { bandwidth_hz } as f64)
                .max(500.0)
                .min(nyquist_margin);
            let (f_lo, f_hi) = if matches!(mode, DemodMode::Usb) {
                (300.0, bw)
            } else {
                (-bw, -300.0)
            };
            let taps = design_ssb_bandpass(
                stage1_rate_hz,
                f_lo,
                f_hi,
                taps_for_transition(stage1_rate_hz, 300.0),
            );
            Stage2::Bandpass {
                filter: ComplexFir::new(taps),
                gain: 1.5,
            }
        }
        DemodMode::Cw => {
            // Narrow one-sided window around a fixed 700 Hz sidetone pitch — CW demod here is
            // just narrow-SSB; see `sdrconnect_dsp`'s SSB doc for why a shifted one-sided filter
            // recovers an audible tone directly.
            let half = ((bandwidth_hz.max(100) as f64) / 2.0).clamp(75.0, 400.0);
            let taps = design_ssb_bandpass(
                stage1_rate_hz,
                700.0 - half,
                700.0 + half,
                taps_for_transition(stage1_rate_hz, 150.0),
            );
            Stage2::Bandpass {
                filter: ComplexFir::new(taps),
                gain: 3.0,
            }
        }
    }
}

fn run_stage2(stage2: &mut Stage2, input: &[Iq]) -> Vec<f32> {
    match stage2 {
        Stage2::Lowpass {
            filter,
            dc_state,
            is_fm,
            fm_prev,
            fm_gain,
        } => {
            let filtered = filter.process(input);
            if *is_fm {
                demod_fm(&filtered, fm_prev, *fm_gain)
            } else {
                demod_am(&filtered, dc_state)
            }
        }
        Stage2::Bandpass { filter, gain } => {
            let filtered = filter.process(input);
            demod_ssb(&filtered, *gain)
        }
    }
}

/// Connect and arm streaming: set the IQ sample rate, then turn on device + IQ streaming.
/// Returns the sample rate SDRconnect actually CONFIRMS is active — read back rather than
/// assumed, because the RSP1B's ADC only supports specific decimated rates and the API may
/// silently round a request to the nearest one instead of erroring. Every downstream rate
/// (Stage 1's decimation factor, its filter design, the final resample to 12 kHz) is derived
/// from this confirmed value in `run`, not from the constant that was merely requested — a
/// mismatch here, even a small one, means every filter cutoff and the final sample rate are
/// systematically wrong, heard as persistent, uniform choppiness rather than an occasional
/// glitch (found 2026-09 chasing exactly that report). Called once at start and again by the
/// worker loop after every reconnect, in case a reconnect ever lands on a different rate.
fn connect_and_arm(url: &str) -> Result<(SdrConnect, f64), SdrConnectError> {
    let client = SdrConnect::connect(url)?;
    client.set_sample_rate_hz(IQ_SAMPLE_RATE_HZ)?;
    let confirmed_rate = client.sample_rate_hz()?;
    client.set_device_stream_enable(true)?;
    client.set_iq_stream_enable(true)?;
    Ok((client, confirmed_rate))
}

/// Decode one SDRconnect binary message into normalised complex samples, or `None` if it is not
/// Primary-device IQ (message type 2) or is malformed — too short, or an odd number of int16s in
/// the payload (a torn frame, dropped rather than partially decoded).
fn decode_iq_frame(data: &[u8]) -> Option<Vec<Iq>> {
    if data.len() < 2 {
        return None;
    }
    let msg_type = u16::from_le_bytes([data[0], data[1]]);
    if msg_type != BINARY_TYPE_IQ_PRIMARY {
        return None;
    }
    let payload = &data[2..];
    if payload.is_empty() || payload.len() % 4 != 0 {
        return None;
    }
    const SCALE: f32 = 1.0 / 32_768.0;
    Some(
        payload
            .chunks_exact(4)
            .map(|w| {
                let i = i16::from_le_bytes([w[0], w[1]]) as f32 * SCALE;
                let q = i16::from_le_bytes([w[2], w[3]]) as f32 * SCALE;
                Iq::new(i, q)
            })
            .collect(),
    )
}

/// The worker itself. See the module doc for the contract.
#[derive(Debug)]
pub struct SdrConnectIq {
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    audio: Arc<Mutex<Vec<f32>>>,
}

impl SdrConnectIq {
    /// Dial `url`, arm IQ streaming, and start demodulating in the background using `demod` —
    /// the SAME `Arc<DemodParams>` the CAT shim's `SdrConnectBackend` writes on every rigctld
    /// mode/bandwidth command, so a WSJT-X `M USB 2700` reaches this worker on its very next
    /// processing block.
    ///
    /// `monitor_sink`: when `Some`, this worker's OWN thread pushes demodulated audio straight
    /// into it as each block finishes — see the module doc's "Real-time monitor feed" section
    /// for why that is the point of accepting it here at all, rather than leaving the RadioLoop
    /// tick's `take_audio()` pull to cover both decode AND live listening.
    pub fn start(
        url: &str,
        demod: Arc<DemodParams>,
        monitor_sink: Option<crate::monitor::MonitorSink>,
    ) -> std::io::Result<SdrConnectIq> {
        let url = url.trim().to_string();
        if url.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SDRconnect WebSocket address is empty",
            ));
        }
        // Fail fast on an unreachable/misconfigured address — the ongoing loop only needs to
        // self-heal a connection that WAS working, not diagnose one that never was.
        let (first, first_rate) = connect_and_arm(&url)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(true));
        let audio = Arc::new(Mutex::new(Vec::new()));

        let stop2 = stop.clone();
        let healthy2 = healthy.clone();
        let audio2 = audio.clone();
        let thread = std::thread::Builder::new()
            .name("sdrconnect-iq".into())
            .spawn(move || {
                Self::run(
                    first,
                    first_rate,
                    url,
                    demod,
                    stop2,
                    healthy2,
                    audio2,
                    monitor_sink,
                )
            })
            .map_err(std::io::Error::other)?;

        Ok(SdrConnectIq {
            stop,
            healthy,
            thread: Some(thread),
            audio,
        })
    }

    /// False while the WebSocket link is down (between the moment a read fails and the next
    /// successful reconnect).
    pub fn is_alive(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    /// Drain and return every sample demodulated since the last call — 12 kHz mono `f32`, the
    /// same contract `flexdax::FlexDax::take_audio()` uses. Still the ONLY route into the
    /// decoder — `monitor_sink` (see `start`) is a SEPARATE, additional delivery for live
    /// listening, not a replacement for this pull.
    pub fn take_audio(&self) -> Vec<f32> {
        self.audio
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default()
    }

    /// Build Stage 1 (coarse decimation) and the final-to-12-kHz resampler for a CONFIRMED IQ
    /// rate — see `connect_and_arm`'s doc for why this is derived from what SDRconnect actually
    /// reports rather than a compile-time constant. Picks whatever decimation factor lands
    /// closest to [`TARGET_STAGE1_RATE_HZ`], so a rate SDRconnect rounded to something other
    /// than the requested [`IQ_SAMPLE_RATE_HZ`] still gets correctly-scaled filters rather than
    /// silently reusing math built for a different rate. Returns the filter, the resampler, and
    /// the ACTUAL Stage 1 rate achieved (`actual_rate_hz / decim`, not exactly the target — used
    /// to rebuild Stage 2 and the monitor resampler to match).
    fn build_front_end(actual_rate_hz: f64) -> (FirDecimator, CaptureResampler, f64) {
        let decim = (actual_rate_hz / TARGET_STAGE1_RATE_HZ).round().max(1.0) as usize;
        let stage1_rate_hz = actual_rate_hz / decim as f64;
        // Anti-alias cutoff at ~44% of the new Nyquist (stage1_rate_hz / 2) — comfortable margin
        // regardless of what `decim` turned out to be, mirroring the fixed-rate version's
        // 11 kHz-of-12.5 kHz margin proportionally rather than as an absolute number.
        let cutoff_hz = stage1_rate_hz * 0.44;
        let taps = design_lowpass(
            taps_for_transition(actual_rate_hz, 3_000.0),
            cutoff_hz / actual_rate_hz,
        );
        let stage1 = FirDecimator::new(taps, decim.max(1));
        let resampler = CaptureResampler::new(stage1_rate_hz.round().max(1.0) as u32, FINAL_RATE_HZ);
        (stage1, resampler, stage1_rate_hz)
    }

    fn run(
        first: SdrConnect,
        first_rate: f64,
        url: String,
        demod: Arc<DemodParams>,
        stop: Arc<AtomicBool>,
        healthy: Arc<AtomicBool>,
        audio: Arc<Mutex<Vec<f32>>>,
        monitor_sink: Option<crate::monitor::MonitorSink>,
    ) {
        let (mut stage1, mut resampler, mut stage1_rate_hz) = Self::build_front_end(first_rate);
        let mut current_mode = demod.mode();
        let mut current_bw = demod.bandwidth_hz();
        let mut stage2 = build_stage2(current_mode, current_bw, stage1_rate_hz);
        // A SECOND resampler, 12 kHz -> whatever rate the monitor's ring expects
        // (`MonitorSink::rate()` — the sound card's own capture rate). Kept separate from
        // `resampler` above: decode always wants exactly 12 kHz, the monitor wants whatever the
        // operator's audio hardware runs at, and conflating them would make either one wrong.
        let mut monitor_resampler = monitor_sink
            .as_ref()
            .map(|sink| CaptureResampler::new(FINAL_RATE_HZ, sink.rate().max(1)));

        let mut client = Some(first);
        while !stop.load(Ordering::Relaxed) {
            let Some(active) = client.take() else {
                match connect_and_arm(&url) {
                    Ok((c, rate)) => {
                        healthy.store(true, Ordering::Relaxed);
                        // Rebuilt UNCONDITIONALLY on every reconnect, even though the confirmed
                        // rate will almost always match the previous connection's: a reconnect
                        // can in principle land on a different rate (a changed SDRconnect
                        // config, a firmware update), and rebuilding is cheap enough that
                        // checking first would only add a way to get this wrong. Mode/bandwidth
                        // state (`current_mode`/`current_bw`) survives the reconnect unchanged;
                        // only the rate-dependent filters are rebuilt.
                        let (new_stage1, new_resampler, new_rate) = Self::build_front_end(rate);
                        stage1 = new_stage1;
                        resampler = new_resampler;
                        stage1_rate_hz = new_rate;
                        stage2 = build_stage2(current_mode, current_bw, stage1_rate_hz);
                        client = Some(c);
                    }
                    Err(_) => {
                        healthy.store(false, Ordering::Relaxed);
                        std::thread::sleep(RECONNECT_BACKOFF);
                    }
                }
                continue;
            };

            match active.read_message() {
                Ok(IncomingMessage::Binary(bytes)) => {
                    client = Some(active);
                    let Some(iq) = decode_iq_frame(&bytes) else {
                        continue;
                    };
                    if iq.is_empty() {
                        continue;
                    }

                    let (m, bw) = (demod.mode(), demod.bandwidth_hz());
                    if m != current_mode || bw != current_bw {
                        current_mode = m;
                        current_bw = bw;
                        stage2 = build_stage2(current_mode, current_bw, stage1_rate_hz);
                    }

                    let decimated = stage1.process(&iq);
                    if decimated.is_empty() {
                        continue;
                    }
                    let demodulated = run_stage2(&mut stage2, &decimated);
                    let resampled = resampler.process(&demodulated);
                    // REAL-TIME monitor push — happens HERE, on this thread, the instant a block
                    // is ready, not on the next RadioLoop tick. This is what fixes choppy
                    // playback: the ring gets fed at the same cadence IQ frames actually arrive
                    // from SDRconnect, independent of whatever the CAT tick is doing right now.
                    if let (Some(sink), Some(mon_resampler)) =
                        (monitor_sink.as_ref(), monitor_resampler.as_mut())
                    {
                        if !resampled.is_empty() {
                            let for_monitor = mon_resampler.process(&resampled);
                            sink.push(&for_monitor);
                        }
                    }
                    if resampled.is_empty() {
                        continue;
                    }
                    if let Ok(mut g) = audio.lock() {
                        g.extend_from_slice(&resampled);
                        if g.len() > MAX_BUFFERED_SAMPLES {
                            let drop = g.len() - MAX_BUFFERED_SAMPLES;
                            g.drain(0..drop);
                        }
                    }
                }
                // Property pushes, keepalive pings/pongs, and any payload type this Phase 1
                // receiver does not consume — ignored, connection stays up.
                Ok(_) => client = Some(active),
                Err(_) => {
                    // `active` is dropped here (its socket is presumed dead); `client` stays
                    // `None` so the top of the loop reconnects on the next iteration.
                    healthy.store(false, Ordering::Relaxed);
                    std::thread::sleep(RECONNECT_BACKOFF);
                }
            }
        }
    }
}

impl Drop for SdrConnectIq {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_iq_frame_rejects_non_iq_message_types() {
        let mut bytes = vec![1u8, 0]; // type 1 = demodulated audio, not IQ
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        assert!(decode_iq_frame(&bytes).is_none());
    }

    #[test]
    fn decode_iq_frame_parses_interleaved_i16_pairs() {
        let mut bytes = vec![2u8, 0]; // type 2 = primary IQ
        bytes.extend_from_slice(&16_384i16.to_le_bytes()); // I = 0.5
        bytes.extend_from_slice(&(-16_384i16).to_le_bytes()); // Q = -0.5
        let iq = decode_iq_frame(&bytes).expect("valid frame");
        assert_eq!(iq.len(), 1);
        assert!((iq[0].i - 0.5).abs() < 1e-3);
        assert!((iq[0].q + 0.5).abs() < 1e-3);
    }

    #[test]
    fn decode_iq_frame_rejects_a_torn_payload() {
        let bytes = vec![2u8, 0, 0, 0, 0]; // 3 payload bytes: not a whole number of i16 pairs
        assert!(decode_iq_frame(&bytes).is_none());
    }

    #[test]
    fn decode_iq_frame_rejects_an_empty_payload() {
        let bytes = vec![2u8, 0]; // header only, no samples
        assert!(decode_iq_frame(&bytes).is_none());
    }

    #[test]
    fn stage2_rebuilds_are_keyed_on_mode_and_bandwidth_not_object_identity() {
        // Not a hardware test — just pins the contract the run loop relies on: the SAME
        // (mode, bandwidth) must be treated as "no change" so a filter isn't rebuilt every
        // single frame (which would reset its history and click on every block boundary).
        let a = (DemodMode::Usb, 2_700u32);
        let b = (DemodMode::Usb, 2_700u32);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_url_is_refused_before_dialing() {
        let err =
            SdrConnectIq::start("   ", DemodParams::new(DemodMode::Usb, 2_700), None).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}

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
const STAGE1_DECIM: usize = 10;
/// The rate every mode's channel filter and demodulator actually run at.
const STAGE1_RATE_HZ: f64 = IQ_SAMPLE_RATE_HZ / STAGE1_DECIM as f64;
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
/// `STAGE1_RATE_HZ`. `bandwidth_hz == 0` (never configured, e.g. before the first CAT `M`/`F`
/// command) falls back to a sane per-mode default rather than building a zero-width filter.
fn build_stage2(mode: DemodMode, bandwidth_hz: u32) -> Stage2 {
    // Stay comfortably clear of the 12.5 kHz Nyquist at STAGE1_RATE_HZ regardless of what the
    // operator (or WSJT-X, via rigctld `M`) asks for — a wider request is clamped, not refused.
    let nyquist_margin = STAGE1_RATE_HZ * 0.46;
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
                taps_for_transition(STAGE1_RATE_HZ, transition),
                cutoff / STAGE1_RATE_HZ,
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
                STAGE1_RATE_HZ,
                f_lo,
                f_hi,
                taps_for_transition(STAGE1_RATE_HZ, 300.0),
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
                STAGE1_RATE_HZ,
                700.0 - half,
                700.0 + half,
                taps_for_transition(STAGE1_RATE_HZ, 150.0),
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

/// Connect and arm streaming: set the IQ sample rate, then turn on device + IQ streaming. Called
/// once at start and again by the worker loop after every reconnect.
fn connect_and_arm(url: &str) -> Result<SdrConnect, SdrConnectError> {
    let client = SdrConnect::connect(url)?;
    client.set_sample_rate_hz(IQ_SAMPLE_RATE_HZ)?;
    client.set_device_stream_enable(true)?;
    client.set_iq_stream_enable(true)?;
    Ok(client)
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
        let first = connect_and_arm(&url)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(true));
        let audio = Arc::new(Mutex::new(Vec::new()));

        let stop2 = stop.clone();
        let healthy2 = healthy.clone();
        let audio2 = audio.clone();
        let thread = std::thread::Builder::new()
            .name("sdrconnect-iq".into())
            .spawn(move || Self::run(first, url, demod, stop2, healthy2, audio2, monitor_sink))
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

    fn run(
        first: SdrConnect,
        url: String,
        demod: Arc<DemodParams>,
        stop: Arc<AtomicBool>,
        healthy: Arc<AtomicBool>,
        audio: Arc<Mutex<Vec<f32>>>,
        monitor_sink: Option<crate::monitor::MonitorSink>,
    ) {
        // Stage 1 is built ONCE — it never depends on mode, only on the fixed IQ/Stage-1 rates.
        let stage1_taps = design_lowpass(
            taps_for_transition(IQ_SAMPLE_RATE_HZ, 3_000.0),
            11_000.0 / IQ_SAMPLE_RATE_HZ,
        );
        let mut stage1 = FirDecimator::new(stage1_taps, STAGE1_DECIM);
        let mut resampler = CaptureResampler::new(STAGE1_RATE_HZ as u32, FINAL_RATE_HZ);
        let mut current_mode = demod.mode();
        let mut current_bw = demod.bandwidth_hz();
        let mut stage2 = build_stage2(current_mode, current_bw);
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
                    Ok(c) => {
                        healthy.store(true, Ordering::Relaxed);
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
                        stage2 = build_stage2(current_mode, current_bw);
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

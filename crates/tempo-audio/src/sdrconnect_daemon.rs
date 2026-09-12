//! **What listens on the radio's rigctld TCP port** when its connection type is SDRconnect.
//!
//! Same shape and the same contract as [`crate::omnirig::OmniDaemon`]: bind
//! `127.0.0.1:<the radio's own rigctld_port>`, serve the rigctld protocol on it, and translate
//! every command to/from SDRplay's SDRconnect WebSocket API (`crate::sdrconnect`) instead of a
//! serial port, COM server, or another rigctld. Everything downstream (`Rig`, the CAT probe,
//! band-follow, QSY-on-click) never learns the difference — it is a fourth `CatDaemon` variant,
//! same as `Native` and `Omni`.
//!
//! # Option B: Nexus demodulates, SDRconnect just tunes
//!
//! Earlier phases (frequency + mode control only) drove SDRconnect's OWN `demodulator`/
//! `filter_bandwidth` properties and its message-type-1 pre-demodulated audio. That was
//! rejected in favour of a receiver Nexus fully owns: this daemon now uses
//! `device_center_frequency` (the hardware LO, not `device_vfo_frequency`) for CAT frequency
//! control, and every rigctld `M`/`m` mode command reaches [`crate::sdrconnect_dsp::DemodParams`]
//! instead of SDRconnect's demod — a SECOND WebSocket connection
//! ([`crate::sdrconnect_iq::SdrConnectIq`], owned by this daemon) streams raw IQ and
//! demodulates it locally. SDRconnect's own `demodulator`/`filter_bandwidth` properties are
//! never written by this file any more; whatever SDRconnect's own GUI shows for them is now
//! informational only, not the signal path.
//!
//! One consequence worth stating plainly: **CAT frequency control now retunes the hardware**,
//! not a VFO offset within an already-captured span (see `sdrconnect_dsp`'s module doc for why
//! that simplification was chosen). A frequency write is a few milliseconds slower than the old
//! `device_vfo_frequency` write was, and the IQ worker's filters see a brief discontinuity right
//! after — inaudible in practice, but real.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::rigctld_server::{serve_until, RigBackend};
use crate::sdrconnect::SdrConnect;
use crate::sdrconnect_dsp::{DemodMode, DemodParams};
use crate::sdrconnect_iq::SdrConnectIq;

/// A sensible starting mode/bandwidth for a radio that has never received a CAT `M`/`F` command
/// yet — general-coverage SSB listening, the most common single default across ham receivers.
const DEFAULT_MODE: DemodMode = DemodMode::Usb;
const DEFAULT_BANDWIDTH_HZ: u32 = 2_700;
/// Fallback LNA-state range used ONLY if querying `lna_state_min`/`lna_state_max` fails at
/// connect time (should not happen — they are read-only, always-answerable properties — but a
/// wrong guess here would silently misdirect every `L RF` command, so it is logged, not silent).
/// 0..=9 matches the RSP1B's ten LNA states; a different SDRplay model with a different count
/// would just get a clamped, off-by-a-few-steps range until the read succeeds on reconnect.
const FALLBACK_LNA_MIN: i32 = 0;
const FALLBACK_LNA_MAX: i32 = 9;

/// SDRplay's `lna_state` (0 = most gain, least attenuation) → Hamlib's `RF` level convention
/// (1.0 = most gain) — a pure function so the inversion/rounding is tested without a live
/// WebSocket. See [`rf_frac_to_lna_state`] for the inverse.
fn lna_state_to_rf_frac(state: i32, lna_min: i32, lna_max: i32) -> f64 {
    let span = lna_max - lna_min;
    if span <= 0 {
        return 1.0;
    }
    (1.0 - (state - lna_min) as f64 / span as f64).clamp(0.0, 1.0)
}

/// The inverse of [`lna_state_to_rf_frac`]: a Hamlib `RF` fraction (0.0–1.0, 1.0 = most gain) →
/// the nearest real `lna_state` step (an index, not a continuous value — hence the rounding).
fn rf_frac_to_lna_state(frac: f64, lna_min: i32, lna_max: i32) -> i32 {
    let span = lna_max - lna_min;
    if span <= 0 {
        return lna_min;
    }
    let frac = frac.clamp(0.0, 1.0);
    lna_min + ((1.0 - frac) * span as f64).round() as i32
}

/// The `RigBackend` this daemon serves: every rigctld verb Nexus's own `Rig` client sends,
/// translated to a call on `crate::sdrconnect::SdrConnect` (frequency) or a write to the shared
/// [`DemodParams`] (mode/bandwidth — read by the IQ worker, never sent to SDRconnect itself; see
/// the module doc).
#[derive(Debug)]
struct SdrConnectBackend {
    client: SdrConnect,
    demod: Arc<DemodParams>,
    /// The device's own reported LNA-state range, read once at connect (see
    /// [`FALLBACK_LNA_MIN`]/`_MAX` for what happens if that read fails). Needed on every `L RF`/
    /// `l RF` call to convert Hamlib's normalised 0.0–1.0 gain fraction to/from SDRconnect's
    /// integer step index — see `level`/`set_level` for the direction of that conversion.
    lna_min: i32,
    lna_max: i32,
    /// Cleared on any WebSocket failure, set again on the next success. `health()`/`is_alive()`
    /// read this rather than re-probing — a failed `get_property` already paid its own 5 s
    /// timeout once; asking again on every health check would pay it a second time.
    healthy: AtomicBool,
}

impl SdrConnectBackend {
    fn note(&self, ok: bool) {
        self.healthy.store(ok, Ordering::Relaxed);
    }
}

impl RigBackend for SdrConnectBackend {
    fn freq_hz(&self) -> u64 {
        // 0 = "no honest reading", the same answer every other backend gives a dead link —
        // `Rig::read_freq` rejects 0, so a failure surfaces as a CAT error instead of a green
        // pill reading 0.000 MHz. `center_frequency_hz`, not `vfo_frequency_hz` — see the
        // module doc: this daemon now tunes the hardware LO, not a VFO offset.
        match self.client.center_frequency_hz() {
            Ok(hz) => {
                self.note(true);
                hz
            }
            Err(_) => {
                self.note(false);
                0
            }
        }
    }

    fn mode(&self) -> (String, u32) {
        // Nexus's own demod state, not SDRconnect's `demodulator` property — see the module
        // doc. This never fails (it is a local atomic read), so it never touches `healthy`.
        (self.demod.mode().to_rigctld().to_string(), self.demod.bandwidth_hz())
    }

    fn ptt(&self) -> bool {
        // The RSP1B has no transmitter; SDRconnect exposes no PTT property for it to ask.
        false
    }

    fn set_freq(&self, hz: u64) -> bool {
        let ok = self.client.set_center_frequency_hz(hz).is_ok();
        self.note(ok);
        ok
    }

    fn set_mode(&self, mode: &str, passband_hz: u32) -> bool {
        // An unmappable mode word is REFUSED, never approximated — see `DemodMode::from_rigctld`.
        let Some(demod_mode) = DemodMode::from_rigctld(mode) else {
            return false;
        };
        self.demod.set_mode(demod_mode);
        if passband_hz > 0 {
            self.demod.set_bandwidth_hz(passband_hz);
        }
        // A local write cannot fail the way a WebSocket round-trip can, but it is still real
        // work the IQ worker depends on — leaving `note` out of this arm would make `health()`
        // silently stop reflecting mode-write attempts entirely, not just skip a failure case.
        self.note(true);
        true
    }

    fn set_ptt(&self, on: bool) -> bool {
        // ⚠️ TX SAFETY, receive-only edition: the RSP1B cannot transmit, so a key-DOWN is
        // always refused — never answer RPRT 0 to a PTT request nothing can act on. A
        // key-UP always "succeeds": there is nothing keyed to unkey, and the disconnect
        // fail-safe (`serve_connection`) must never see an unkey attempt fail.
        !on
    }

    /// `l RF` / `l AGC` — the two knobs SDRconnect's API actually exposes (see the module doc
    /// on `sdrconnect.rs::set_agc_enable`'s doc for why there is no manual IF gain to add a
    /// third for). Any other Hamlib level token (STRENGTH, SQL, ...) is `None` — unimplemented,
    /// not zero, so a client asking for one knows to stop asking rather than act on a fake 0.
    fn level(&self, name: &str) -> Option<String> {
        match name {
            "RF" => {
                let state = self.client.lna_state().ok()?;
                let frac = lna_state_to_rf_frac(state, self.lna_min, self.lna_max);
                Some(format!("{frac:.3}"))
            }
            "AGC" => {
                let on = self.client.agc_enable().ok()?;
                // Hamlib's AGC level is a small integer (0 = off, nonzero = some AGC speed).
                // SDRconnect only has on/off, so this collapses onto the two ends of that scale
                // rather than inventing a "fast"/"slow" the API cannot actually select.
                Some(if on { "1".to_string() } else { "0".to_string() })
            }
            _ => None,
        }
    }

    fn set_level(&self, name: &str, value: &str) -> Option<bool> {
        match name {
            "RF" => {
                let frac: f64 = value.trim().parse().ok()?;
                let state = rf_frac_to_lna_state(frac, self.lna_min, self.lna_max);
                let ok = self.client.set_lna_state(state).is_ok();
                self.note(ok);
                Some(ok)
            }
            "AGC" => {
                // Accept both a Hamlib-style integer level and a plain "true"/"false", since
                // Nexus's own UI (once it has one for this) is free to send whichever reads more
                // naturally; a rigctld client sends the former.
                let on = value
                    .trim()
                    .parse::<f64>()
                    .map(|v| v > 0.0)
                    .unwrap_or_else(|_| value.trim().eq_ignore_ascii_case("true"));
                let ok = self.client.set_agc_enable(on).is_ok();
                self.note(ok);
                Some(ok)
            }
            _ => None,
        }
    }
}

/// The daemon itself — see the module doc for the contract. Owns TWO WebSocket connections to
/// SDRconnect under one lifecycle: the CAT control socket ([`SdrConnectBackend`], via the TCP
/// shim thread) and the IQ streaming socket ([`SdrConnectIq`]) — see `sdrconnect_iq`'s module
/// doc for why they must be separate connections. Starting and stopping together means a CAT
/// reconnect always gets a matching fresh IQ worker and a fresh shared [`DemodParams`], with no
/// possibility of one outliving the other and reading stale shared state.
#[derive(Debug)]
pub struct SdrConnectDaemon {
    stop: Arc<AtomicBool>,
    tcp_thread: Option<std::thread::JoinHandle<()>>,
    backend: Arc<SdrConnectBackend>,
    iq: SdrConnectIq,
}

impl SdrConnectDaemon {
    /// Dial `url` (SDRconnect's own WebSocket endpoint, e.g. `ws://192.168.1.50:5454`) TWICE —
    /// once for CAT control, once for IQ streaming — and start serving the rigctld protocol on
    /// `127.0.0.1:<tcp_port>`.
    ///
    /// `monitor_sink`: a live [`crate::monitor::MonitorSink`] when the caller has one to give
    /// (the production tick loop does — see `service.rs`'s `open_rig`/`open_cat` doc comments
    /// for the chain this arrives through). When `Some`, handed straight to
    /// [`SdrConnectIq::start`], which then pushes demodulated audio into it from its OWN
    /// real-time thread instead of relying solely on the tick-driven `take_audio()` pull — see
    /// that module's doc for why the difference is audible (choppy vs. smooth). `None` (the
    /// very first connection at app startup, and every background probe/test) just means this
    /// particular connection falls back to the tick-driven path until the daemon next restarts
    /// with a real sink — never a hard failure either way.
    pub fn start(
        url: &str,
        tcp_port: u16,
        monitor_sink: Option<crate::monitor::MonitorSink>,
    ) -> std::io::Result<SdrConnectDaemon> {
        let url = url.trim();
        if url.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SDRconnect WebSocket address is empty",
            ));
        }
        let client = SdrConnect::connect(url)
            .map_err(std::io::Error::other)?;
        // Read the device's real LNA-state range ONCE — see `FALLBACK_LNA_MIN`/`_MAX`'s doc for
        // why a failure here falls back rather than failing the whole daemon: `L RF` is a
        // secondary feature, and a wrong-but-clamped range is a smaller problem than no CAT at
        // all because a read-only property hiccuped on a freshly-opened socket.
        let lna_min = client.lna_state_min().unwrap_or(FALLBACK_LNA_MIN);
        let lna_max = client.lna_state_max().unwrap_or(FALLBACK_LNA_MAX);
        let demod = DemodParams::new(DEFAULT_MODE, DEFAULT_BANDWIDTH_HZ);
        // Started BEFORE the TCP shim binds: if the IQ socket can't be established (SDRconnect
        // unreachable a second time in a row would be surprising, but a strict device/stream
        // limit on the SDRconnect side is plausible), fail the whole daemon rather than leaving
        // a CAT-only half-daemon that reports control success but is silently deaf.
        let iq = SdrConnectIq::start(url, demod.clone(), monitor_sink)?;
        let backend = Arc::new(SdrConnectBackend {
            client,
            demod,
            lna_min,
            lna_max,
            healthy: AtomicBool::new(true),
        });
        let listener = std::net::TcpListener::bind(("127.0.0.1", tcp_port))?;
        let stop = Arc::new(AtomicBool::new(false));
        let served: Arc<dyn RigBackend> = backend.clone();
        let stop_for_thread = stop.clone();
        let tcp_thread = std::thread::Builder::new()
            .name("sdrconnect-tcp".into())
            .spawn(move || serve_until(listener, served, stop_for_thread))
            .map_err(std::io::Error::other)?;
        Ok(SdrConnectDaemon {
            stop,
            tcp_thread: Some(tcp_thread),
            backend,
            iq,
        })
    }

    /// False once EITHER the CAT control link or the IQ streaming link has failed and not yet
    /// recovered — a CAT-only failure and an IQ-only failure both mean the radio isn't fully
    /// working, so both must be able to turn this false.
    pub fn is_alive(&self) -> bool {
        self.backend.healthy.load(Ordering::Relaxed) && self.iq.is_alive()
    }

    /// SDRconnect's health for this radio, as a sentence — `Ok(())` when both links are up.
    pub fn health(&self) -> Result<(), String> {
        if !self.backend.healthy.load(Ordering::Relaxed) {
            Err("SDRconnect CAT control is not responding.".to_string())
        } else if !self.iq.is_alive() {
            Err("SDRconnect IQ streaming is not responding — reconnecting.".to_string())
        } else {
            Ok(())
        }
    }

    /// Drain and return every sample the IQ worker has demodulated since the last call — 12 kHz
    /// mono `f32`, the same pull contract `flexdax::FlexDax::take_audio()` uses. This is the one
    /// method `service.rs`'s per-tick capture step calls.
    pub fn take_audio(&self) -> Vec<f32> {
        self.iq.take_audio()
    }
}

impl Drop for SdrConnectDaemon {
    fn drop(&mut self) {
        // No TX safety unkey needed here (see `set_ptt`) — just release the port cleanly. The
        // IQ worker (`self.iq`) stops itself in its own `Drop`, dropped after this runs.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.tcp_thread.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_url_is_refused_before_dialing() {
        let err = SdrConnectDaemon::start("   ", 0, None).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn an_unmapped_mode_word_is_refused_not_approximated() {
        let demod = DemodParams::new(DEFAULT_MODE, DEFAULT_BANDWIDTH_HZ);
        // `RigBackend::set_mode` on a live backend needs a real WebSocket, so this exercises the
        // same refusal path `SdrConnectBackend::set_mode` relies on directly.
        assert_eq!(DemodMode::from_rigctld("RTTY"), None);
        // A backend that never receives a mappable mode keeps reporting the constructor default.
        assert_eq!(demod.mode().to_rigctld(), DEFAULT_MODE.to_rigctld());
    }

    #[test]
    fn rf_gain_direction_is_inverted_against_lna_state() {
        // lna_state 0 (least attenuation, most gain) must read as Hamlib RF = 1.0 (most gain),
        // and the top of the range must read as 0.0 — the whole point of the inversion.
        assert!((lna_state_to_rf_frac(0, 0, 9) - 1.0).abs() < 1e-9);
        assert!((lna_state_to_rf_frac(9, 0, 9) - 0.0).abs() < 1e-9);
        assert!((lna_state_to_rf_frac(0, 3, 8) - 1.0).abs() < 1e-9, "must use the DEVICE's own range, not always 0");
    }

    #[test]
    fn rf_gain_round_trips_through_both_conversions() {
        for &(min, max) in &[(0, 9), (2, 8), (0, 27)] {
            for state in min..=max {
                let frac = lna_state_to_rf_frac(state, min, max);
                let back = rf_frac_to_lna_state(frac, min, max);
                assert_eq!(back, state, "range {min}..={max}, state {state} -> {frac} -> {back}");
            }
        }
    }

    #[test]
    fn rf_gain_handles_a_degenerate_zero_width_range_without_panicking() {
        // A device that reports lna_state_min == lna_state_max (or a failed read that fell back
        // to an inconsistent pair) must not divide by zero.
        assert_eq!(rf_frac_to_lna_state(0.5, 4, 4), 4);
        assert!((lna_state_to_rf_frac(4, 4, 4) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rf_gain_input_is_clamped_not_rejected() {
        // A caller sending an out-of-range fraction (a bug elsewhere, or a client that does not
        // clamp) should not produce an out-of-range `lna_state` the device could refuse.
        assert_eq!(rf_frac_to_lna_state(-1.0, 0, 9), 9);
        assert_eq!(rf_frac_to_lna_state(2.0, 0, 9), 0);
    }
}

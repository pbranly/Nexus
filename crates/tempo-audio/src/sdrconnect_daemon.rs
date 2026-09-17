//! **What listens on the radio's rigctld TCP port** when its connection type is SDRconnect.
//!
//! Same shape and the same contract as [`crate::omnirig::OmniDaemon`]: bind
//! `127.0.0.1:<the radio's own rigctld_port>`, serve the rigctld protocol on it, and translate
//! every command to/from SDRplay's SDRconnect WebSocket API (`crate::sdrconnect`) instead of a
//! serial port, COM server, or another rigctld. Everything downstream (`Rig`, the CAT probe,
//! band-follow, QSY-on-click) never learns the difference — it is a fourth `CatDaemon` variant,
//! same as `Native` and `Omni`.
//!
//! # Phase 1: control only
//!
//! This is deliberately the SIMPLE version: frequency, mode and filter bandwidth are read and
//! written straight through to SDRconnect's OWN `device_vfo_frequency`/`demodulator`/
//! `filter_bandwidth` properties — SDRconnect keeps doing its own demodulation and producing its
//! own audio (heard however the operator has SDRconnect itself configured to output it; this
//! phase does not touch that). No IQ streaming, no local DSP, no headphone-monitor integration.
//! An earlier, much more ambitious version of this file had Nexus do its own local
//! software-defined demodulation from raw IQ — it caused a long chain of hard-to-diagnose
//! real-time audio bugs (see the project history) and was abandoned in favour of getting this
//! simpler, lower-risk version solid first. RF gain (`lna_state`) and AGC (`agc_enable`) are
//! wired through as rigctld levels ("RF"/"AGC") since SDRconnect's API exposes them and Hamlib
//! already has tokens for both.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::rigctld_server::{serve_until, RigBackend};
use crate::sdrconnect::SdrConnect;

/// Fallback LNA-state range used ONLY if querying `lna_state_min`/`lna_state_max` fails at
/// connect time (read-only, always-answerable properties — should not happen, but a wrong guess
/// here would silently misdirect every `L RF` command). 0..=9 matches the RSP1B's ten LNA
/// states; a different SDRplay model would just get a clamped range until the read succeeds on
/// reconnect.
const FALLBACK_LNA_MIN: i32 = 0;
const FALLBACK_LNA_MAX: i32 = 9;

/// SDRplay's `lna_state` (0 = most gain, least attenuation) → Hamlib's `RF` level convention
/// (1.0 = most gain) — a pure function so the inversion/rounding is unit-tested without a live
/// WebSocket.
fn lna_state_to_rf_frac(state: i32, lna_min: i32, lna_max: i32) -> f64 {
    let span = lna_max - lna_min;
    if span <= 0 {
        return 1.0;
    }
    (1.0 - (state - lna_min) as f64 / span as f64).clamp(0.0, 1.0)
}

/// The inverse of [`lna_state_to_rf_frac`]: a Hamlib `RF` fraction (0.0-1.0, 1.0 = most gain) →
/// the nearest real `lna_state` step.
fn rf_frac_to_lna_state(frac: f64, lna_min: i32, lna_max: i32) -> i32 {
    let span = lna_max - lna_min;
    if span <= 0 {
        return lna_min;
    }
    let frac = frac.clamp(0.0, 1.0);
    lna_min + ((1.0 - frac) * span as f64).round() as i32
}

/// A rigctld mode word → SDRconnect's `demodulator` value. SDRconnect has one CW and one
/// AM/SAM pair and no DATA-submode distinction, so every rigctld DATA word (`PKTUSB`,
/// `DATA-U`, ...) collapses onto plain `USB`/`LSB` — there is nothing on the SDR side to select
/// differently. `None` for a word SDRconnect cannot represent, refused rather than approximated.
fn rigctld_to_sdr_demod(word: &str) -> Option<&'static str> {
    Some(match word.trim().to_ascii_uppercase().as_str() {
        "USB" | "PKTUSB" | "DATA-U" | "PKT-U" | "USB-D" => "USB",
        "LSB" | "PKTLSB" | "DATA-L" | "PKT-L" | "LSB-D" => "LSB",
        "CW" | "CW-U" | "CWU" | "CWR" | "CW-L" | "CWL" => "CW",
        "AM" => "AM",
        "FM" | "PKTFM" | "FM-D" | "PKT-FM" => "NFM",
        "WFM" => "WFM",
        _ => return None,
    })
}

/// SDRconnect's `demodulator` value → the rigctld mode word reported back. `SAM` reports as
/// plain `AM` (rigctld has no synchronous-AM word); anything unrecognised reports as `USB`
/// rather than blank, the "never report an empty mode" rule every backend follows.
fn sdr_demod_to_rigctld(demod: &str) -> &'static str {
    match demod.trim().to_ascii_uppercase().as_str() {
        "USB" => "USB",
        "LSB" => "LSB",
        "CW" => "CW",
        "AM" | "SAM" => "AM",
        "NFM" => "FM",
        "WFM" => "WFM",
        _ => "USB",
    }
}

/// The `RigBackend` this daemon serves: every rigctld verb Nexus's own `Rig` client sends,
/// translated to a call on `crate::sdrconnect::SdrConnect`.
#[derive(Debug)]
struct SdrConnectBackend {
    /// The URL to reconnect to — a socket that has failed once cannot be "fixed", a NEW one has
    /// to be dialled. Behind a `Mutex` (not a plain field) so a `&self` trait method — every
    /// `RigBackend` method is `&self`, never `&mut self` — can still replace it after a
    /// reconnect.
    url: String,
    client: std::sync::Mutex<SdrConnect>,
    lna_min: i32,
    lna_max: i32,
    /// Cleared on any WebSocket failure, set again on the next success. `health()`/`is_alive()`
    /// read this rather than re-probing.
    healthy: AtomicBool,
}

impl SdrConnectBackend {
    fn note(&self, ok: bool) {
        self.healthy.store(ok, Ordering::Relaxed);
    }

    /// Run `f` against the live socket; on failure, dial ONE fresh connection and retry `f` once
    /// more before giving up. Without this, a transient WebSocket hiccup left `client`
    /// permanently broken until `RadioLoop`'s much coarser daemon-rebuild recovery eventually
    /// noticed — several failed probes and a backoff, torn down and rebuilt from scratch. A
    /// hiccup now heals in-place, in the time one handshake takes.
    fn with_client<T>(
        &self,
        f: impl Fn(&SdrConnect) -> Result<T, crate::sdrconnect::SdrConnectError>,
    ) -> Option<T> {
        {
            let guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
            if let Ok(v) = f(&guard) {
                return Some(v);
            }
        }
        let fresh = SdrConnect::connect(&self.url).ok()?;
        let mut guard = self.client.lock().unwrap_or_else(|e| e.into_inner());
        *guard = fresh;
        f(&guard).ok()
    }
}

impl RigBackend for SdrConnectBackend {
    fn freq_hz(&self) -> u64 {
        // 0 = "no honest reading" — `Rig::read_freq` rejects 0, so a failure surfaces as a CAT
        // error instead of a green pill reading 0.000 MHz.
        match self.with_client(|c| c.vfo_frequency_hz()) {
            Some(hz) => {
                self.note(true);
                hz
            }
            None => {
                self.note(false);
                0
            }
        }
    }

    fn mode(&self) -> (String, u32) {
        let demod = self.with_client(|c| c.demodulator());
        self.note(demod.is_some());
        let mode = demod
            .as_deref()
            .map(sdr_demod_to_rigctld)
            .unwrap_or("USB")
            .to_string();
        // Best-effort: a stale/unknown passband must not turn an otherwise-good mode read into
        // a failure reported as "no CAT".
        let passband_hz = self
            .with_client(|c| c.filter_bandwidth_hz())
            .unwrap_or(0);
        (mode, passband_hz)
    }

    fn ptt(&self) -> bool {
        // The RSP1B has no transmitter; SDRconnect exposes no PTT property for it to ask.
        false
    }

    fn set_freq(&self, hz: u64) -> bool {
        let ok = self.with_client(|c| c.set_vfo_frequency_hz(hz)).is_some();
        self.note(ok);
        ok
    }

    fn set_mode(&self, mode: &str, passband_hz: u32) -> bool {
        // An unmappable mode word is REFUSED, never approximated.
        let Some(demod) = rigctld_to_sdr_demod(mode) else {
            return false;
        };
        let mut ok = self.with_client(|c| c.set_demodulator(demod)).is_some();
        if ok && passband_hz > 0 {
            ok = self
                .with_client(|c| c.set_filter_bandwidth_hz(passband_hz))
                .is_some();
        }
        self.note(ok);
        ok
    }

    fn set_ptt(&self, on: bool) -> bool {
        // ⚠️ TX SAFETY, receive-only edition: the RSP1B cannot transmit, so a key-DOWN is
        // always refused — never answer RPRT 0 to a PTT request nothing can act on. A key-UP
        // always "succeeds": there is nothing keyed to unkey.
        !on
    }

    /// `l RF` / `l AGC` — the two knobs SDRconnect's API exposes beyond mode/frequency. Any
    /// other Hamlib level token is `None` — unimplemented, not zero.
    fn level(&self, name: &str) -> Option<String> {
        match name {
            "RF" => {
                let state = self.with_client(|c| c.lna_state())?;
                let frac = lna_state_to_rf_frac(state, self.lna_min, self.lna_max);
                Some(format!("{frac:.3}"))
            }
            "AGC" => {
                let on = self.with_client(|c| c.agc_enable())?;
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
                let ok = self.with_client(|c| c.set_lna_state(state)).is_some();
                self.note(ok);
                Some(ok)
            }
            "AGC" => {
                let on = value
                    .trim()
                    .parse::<f64>()
                    .map(|v| v > 0.0)
                    .unwrap_or_else(|_| value.trim().eq_ignore_ascii_case("true"));
                let ok = self.with_client(|c| c.set_agc_enable(on)).is_some();
                self.note(ok);
                Some(ok)
            }
            _ => None,
        }
    }
}

/// The daemon itself — see the module doc for the contract.
#[derive(Debug)]
pub struct SdrConnectDaemon {
    stop: Arc<AtomicBool>,
    tcp_thread: Option<std::thread::JoinHandle<()>>,
    backend: Arc<SdrConnectBackend>,
}

impl SdrConnectDaemon {
    /// Dial `url` (SDRconnect's own WebSocket endpoint, e.g. `ws://192.168.1.50:5454`) and start
    /// serving the rigctld protocol on `127.0.0.1:<tcp_port>`.
    pub fn start(url: &str, tcp_port: u16) -> std::io::Result<SdrConnectDaemon> {
        let url = url.trim();
        if url.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SDRconnect WebSocket address is empty",
            ));
        }
        let client = SdrConnect::connect(url).map_err(std::io::Error::other)?;
        let lna_min = client.lna_state_min().unwrap_or(FALLBACK_LNA_MIN);
        let lna_max = client.lna_state_max().unwrap_or(FALLBACK_LNA_MAX);
        let backend = Arc::new(SdrConnectBackend {
            url: url.to_string(),
            client: std::sync::Mutex::new(client),
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
        })
    }

    /// False while the CAT WebSocket link is down (between a failed call and the next
    /// successful reconnect via `with_client`).
    pub fn is_alive(&self) -> bool {
        self.backend.healthy.load(Ordering::Relaxed)
    }

    /// SDRconnect's health for this radio, as a sentence — `Ok(())` when it is answering.
    pub fn health(&self) -> Result<(), String> {
        if self.is_alive() {
            Ok(())
        } else {
            Err("SDRconnect is not responding.".to_string())
        }
    }
}

impl Drop for SdrConnectDaemon {
    fn drop(&mut self) {
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
    fn rigctld_data_submodes_collapse_to_plain_sideband() {
        assert_eq!(rigctld_to_sdr_demod("PKTUSB"), Some("USB"));
        assert_eq!(rigctld_to_sdr_demod("PKTLSB"), Some("LSB"));
        assert_eq!(rigctld_to_sdr_demod("DATA-U"), Some("USB"));
        assert_eq!(rigctld_to_sdr_demod("DATA-L"), Some("LSB"));
    }

    #[test]
    fn rigctld_fm_words_map_to_narrow_fm() {
        assert_eq!(rigctld_to_sdr_demod("FM"), Some("NFM"));
        assert_eq!(rigctld_to_sdr_demod("PKTFM"), Some("NFM"));
    }

    #[test]
    fn unmappable_rigctld_mode_is_refused_not_approximated() {
        assert_eq!(rigctld_to_sdr_demod("RTTY"), None);
        assert_eq!(rigctld_to_sdr_demod(""), None);
    }

    #[test]
    fn sdr_demod_reports_never_go_blank() {
        assert_eq!(sdr_demod_to_rigctld("SAM"), "AM");
        assert_eq!(sdr_demod_to_rigctld("bogus"), "USB");
        assert_eq!(sdr_demod_to_rigctld("WFM"), "WFM");
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
        assert_eq!(rf_frac_to_lna_state(0.5, 4, 4), 4);
        assert!((lna_state_to_rf_frac(4, 4, 4) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_url_is_refused_before_dialing() {
        let err = SdrConnectDaemon::start("   ", 0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}

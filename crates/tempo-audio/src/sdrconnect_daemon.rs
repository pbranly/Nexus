//! **What listens on the radio's rigctld TCP port** when its connection type is SDRconnect.
//!
//! Same shape and the same contract as [`crate::omnirig::OmniDaemon`]: bind
//! `127.0.0.1:<the radio's own rigctld_port>`, serve the rigctld protocol on it, and translate
//! every command to/from SDRplay's SDRconnect WebSocket API (`crate::sdrconnect`) instead of a
//! serial port, COM server, or another rigctld. Everything downstream (`Rig`, the CAT probe,
//! band-follow, QSY-on-click) never learns the difference — it is a fourth `CatDaemon` variant,
//! same as `Native` and `Omni`.
//!
//! Phase 1 only: frequency, demodulator mode and filter bandwidth. No PTT (the RSP1B is a
//! receive-only SDR) and no IQ/audio/spectrum streaming — `crate::sdrconnect::SdrConnect`
//! deliberately does not enable those either yet.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::rigctld_server::{serve_until, RigBackend};
use crate::sdrconnect::SdrConnect;

/// A rigctld mode word → SDRconnect's `demodulator` value.
///
/// SDRconnect has one CW and one AM/SAM pair and no DATA-submode distinction, so every
/// rigctld DATA word (`PKTUSB`, `DATA-U`, ...) collapses onto plain `USB`/`LSB` — there is
/// nothing on the SDR side for it to select differently. `None` for a word SDRconnect cannot
/// represent, refused rather than approximated, matching `OmniMode::from_rigctld`'s rule: a
/// silently-wrong emission is worse than a refused command.
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

/// SDRconnect's `demodulator` value → the rigctld mode word Nexus (and any Hamlib client)
/// reads back. `SAM` reports as plain `AM` — rigctld has no synchronous-AM word — and any
/// unrecognised value reports as `USB` rather than an empty string, the same "never blank"
/// rule `OmniMode::mode` follows when OmniRig has not reported yet.
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
struct SdrConnectBackend {
    client: SdrConnect,
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
        // pill reading 0.000 MHz.
        match self.client.vfo_frequency_hz() {
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
        let demod = self.client.demodulator();
        self.note(demod.is_ok());
        let mode = demod
            .as_deref()
            .map(sdr_demod_to_rigctld)
            .unwrap_or("USB")
            .to_string();
        // Filter bandwidth is read best-effort: a stale/unknown passband must not turn an
        // otherwise-good mode read into a failure Nexus reports as "no CAT".
        let passband_hz = self.client.filter_bandwidth_hz().unwrap_or(0);
        (mode, passband_hz)
    }

    fn ptt(&self) -> bool {
        // The RSP1B has no transmitter; SDRconnect exposes no PTT property for it to ask.
        false
    }

    fn set_freq(&self, hz: u64) -> bool {
        let ok = self.client.set_vfo_frequency_hz(hz).is_ok();
        self.note(ok);
        ok
    }

    fn set_mode(&self, mode: &str, passband_hz: u32) -> bool {
        // An unmappable mode word is REFUSED, never approximated — see `rigctld_to_sdr_demod`.
        let Some(demod) = rigctld_to_sdr_demod(mode) else {
            return false;
        };
        let mut ok = self.client.set_demodulator(demod).is_ok();
        if ok && passband_hz > 0 {
            ok = self.client.set_filter_bandwidth_hz(passband_hz).is_ok();
        }
        self.note(ok);
        ok
    }

    fn set_ptt(&self, on: bool) -> bool {
        // ⚠️ TX SAFETY, receive-only edition: the RSP1B cannot transmit, so a key-DOWN is
        // always refused — never answer RPRT 0 to a PTT request nothing can act on. A
        // key-UP always "succeeds": there is nothing keyed to unkey, and the disconnect
        // fail-safe (`serve_connection`) must never see an unkey attempt fail.
        !on
    }
}

/// The daemon itself — see the module doc for the contract.
pub struct SdrConnectDaemon {
    stop: Arc<AtomicBool>,
    tcp_thread: Option<std::thread::JoinHandle<()>>,
    backend: Arc<SdrConnectBackend>,
}

impl SdrConnectDaemon {
    /// Dial `url` (SDRconnect's own WebSocket endpoint, e.g. `ws://192.168.1.50:5454`) and
    /// start serving the rigctld protocol on `127.0.0.1:<tcp_port>`.
    pub fn start(url: &str, tcp_port: u16) -> std::io::Result<SdrConnectDaemon> {
        let url = url.trim();
        if url.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SDRconnect WebSocket address is empty",
            ));
        }
        let client = SdrConnect::connect(url).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        let backend = Arc::new(SdrConnectBackend {
            client,
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

    /// False once the WebSocket link has failed a call and not yet recovered.
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
        // No TX safety unkey needed here (see `set_ptt`) — just release the port cleanly.
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
    fn empty_url_is_refused_before_dialing() {
        let err = SdrConnectDaemon::start("   ", 0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}

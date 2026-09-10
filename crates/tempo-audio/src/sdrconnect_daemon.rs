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

/// The `RigBackend` this daemon serves: every rigctld verb Nexus's own `Rig` client sends,
/// translated to a call on `crate::sdrconnect::SdrConnect` (frequency) or a write to the shared
/// [`DemodParams`] (mode/bandwidth — read by the IQ worker, never sent to SDRconnect itself; see
/// the module doc).
#[derive(Debug)]
struct SdrConnectBackend {
    client: SdrConnect,
    demod: Arc<DemodParams>,
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
    pub fn start(url: &str, tcp_port: u16) -> std::io::Result<SdrConnectDaemon> {
        let url = url.trim();
        if url.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SDRconnect WebSocket address is empty",
            ));
        }
        let client = SdrConnect::connect(url)
            .map_err(std::io::Error::other)?;
        let demod = DemodParams::new(DEFAULT_MODE, DEFAULT_BANDWIDTH_HZ);
        // Started BEFORE the TCP shim binds: if the IQ socket can't be established (SDRconnect
        // unreachable a second time in a row would be surprising, but a strict device/stream
        // limit on the SDRconnect side is plausible), fail the whole daemon rather than leaving
        // a CAT-only half-daemon that reports control success but is silently deaf.
        let iq = SdrConnectIq::start(url, demod.clone())?;
        let backend = Arc::new(SdrConnectBackend {
            client,
            demod,
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
        let err = SdrConnectDaemon::start("   ", 0).unwrap_err();
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
}

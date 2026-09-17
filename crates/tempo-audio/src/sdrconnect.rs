//! A minimal blocking client for SDRplay's **SDRconnect WebSocket API** (spec v1.0.3) —
//! control only. This module deliberately does NOT stream IQ or audio: it exists to let
//! `sdrconnect_daemon` read and write a handful of device properties (frequency, mode, filter
//! bandwidth, RF gain, AGC) over one long-lived WebSocket connection, the same way `omnirig`
//! wraps a COM link for OmniRig.
//!
//! # Wire format
//!
//! Every message either way is one JSON object:
//! ```json
//! {"event_type": "set_property", "property": "device_center_frequency", "value": "14074000"}
//! ```
//! A `get_property` is answered with a `get_property_response` carrying the same `property` and
//! the current `value` (always a string — numbers are sent and parsed as decimal text, not JSON
//! numbers). Streaming toggles (`device_stream_enable`, `iq_stream_enable`, `audio_stream_enable`)
//! use the same envelope with an empty `property` — not used by this module yet, but the
//! constants are here so a future IQ/audio phase does not have to reinvent the envelope.

use std::fmt;
use std::net::TcpStream;
use std::time::Duration;

use tungstenite::{connect, Message, WebSocket};

type Ws = WebSocket<tungstenite::stream::MaybeTlsStream<TcpStream>>;

pub mod event {
    pub const SET_PROPERTY: &str = "set_property";
    pub const GET_PROPERTY: &str = "get_property";
    pub const GET_PROPERTY_RESPONSE: &str = "get_property_response";
}

/// SDRconnect property names this module reads/writes. Not exhaustive — only what
/// `sdrconnect_daemon`'s CAT shim currently needs.
pub mod property {
    /// The hardware LO / tuned frequency. In Phase 1 (this module), Nexus never sets a
    /// different `device_center_frequency` — it drives `device_vfo_frequency` and lets
    /// SDRconnect's own demodulator do the work, exactly like any other CAT rig it does not
    /// otherwise touch. `device_center_frequency` is still readable here for completeness/future
    /// use, but the CAT shim's `freq_hz`/`set_freq` use `VFO_FREQUENCY` below.
    pub const CENTER_FREQUENCY: &str = "device_center_frequency";
    pub const VFO_FREQUENCY: &str = "device_vfo_frequency";
    pub const SAMPLE_RATE: &str = "device_sample_rate";
    pub const FILTER_BANDWIDTH: &str = "filter_bandwidth";
    pub const DEMODULATOR: &str = "demodulator";
    pub const LNA_STATE: &str = "lna_state";
    pub const LNA_STATE_MIN: &str = "lna_state_min";
    pub const LNA_STATE_MAX: &str = "lna_state_max";
    pub const AGC_ENABLE: &str = "agc_enable";
}

#[derive(Debug, Clone, PartialEq)]
pub enum SdrConnectError {
    WebSocket(String),
    Json(String),
    Timeout(String),
    Protocol(String),
}

impl fmt::Display for SdrConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SdrConnectError::WebSocket(e) => write!(f, "SDRconnect WebSocket error: {e}"),
            SdrConnectError::Json(e) => write!(f, "SDRconnect malformed message: {e}"),
            SdrConnectError::Timeout(e) => write!(f, "SDRconnect timed out: {e}"),
            SdrConnectError::Protocol(e) => write!(f, "SDRconnect protocol error: {e}"),
        }
    }
}

impl std::error::Error for SdrConnectError {}

impl From<tungstenite::Error> for SdrConnectError {
    fn from(e: tungstenite::Error) -> Self {
        SdrConnectError::WebSocket(e.to_string())
    }
}

/// One control connection to SDRconnect. `Debug` is hand-written (not derived) because the
/// underlying `WebSocket`/`TcpStream` are not `Debug` — this just names the type, it does not
/// dump the socket state.
pub struct SdrConnect {
    socket: std::sync::Mutex<Ws>,
    timeout: Duration,
}

impl fmt::Debug for SdrConnect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SdrConnect")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl SdrConnect {
    /// Dial `url` (e.g. `ws://192.168.1.50:5454`). A fresh connection — SDRconnect hands back a
    /// consistent device state on connect; no explicit handshake beyond the WebSocket upgrade is
    /// needed before `get_property`/`set_property` work.
    pub fn connect(url: &str) -> Result<Self, SdrConnectError> {
        let (socket, _response) = connect(url)?;
        Ok(Self {
            socket: std::sync::Mutex::new(socket),
            timeout: Duration::from_secs(5),
        })
    }

    /// Override the default 5 s property round-trip timeout.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn send_json(&self, socket: &mut Ws, value: &serde_json::Value) -> Result<(), SdrConnectError> {
        let text = serde_json::to_string(value).map_err(|e| SdrConnectError::Json(e.to_string()))?;
        socket.send(Message::Text(text.into()))?;
        Ok(())
    }

    pub fn set_property(&self, property: &str, value: impl Into<String>) -> Result<(), SdrConnectError> {
        let mut socket = self.socket.lock().unwrap_or_else(|e| e.into_inner());
        let msg = serde_json::json!({
            "event_type": event::SET_PROPERTY,
            "property": property,
            "value": value.into(),
        });
        self.send_json(&mut socket, &msg)
    }

    /// Send `get_property` and block for the matching `get_property_response`, skipping (not
    /// erroring on) any other message that arrives first — SDRconnect can interleave unrelated
    /// property-changed pushes on this same connection.
    pub fn get_property(&self, property: impl Into<String>) -> Result<String, SdrConnectError> {
        let property = property.into();
        let mut socket = self.socket.lock().unwrap_or_else(|e| e.into_inner());
        let msg = serde_json::json!({
            "event_type": event::GET_PROPERTY,
            "property": &property,
            "value": "",
        });
        self.send_json(&mut socket, &msg)?;

        let deadline = std::time::Instant::now() + self.timeout;
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(SdrConnectError::Timeout(format!(
                    "no response for property {property:?} within {:?}",
                    self.timeout
                )));
            }
            let incoming = socket.read()?;
            let text = match incoming {
                Message::Text(t) => t,
                Message::Binary(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {
                    continue
                }
                Message::Close(_) => {
                    return Err(SdrConnectError::Protocol(
                        "connection closed while waiting for a property response".into(),
                    ))
                }
            };
            let parsed: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| SdrConnectError::Json(e.to_string()))?;
            if parsed.get("event_type").and_then(|v| v.as_str()) != Some(event::GET_PROPERTY_RESPONSE) {
                continue;
            }
            if parsed.get("property").and_then(|v| v.as_str()) != Some(property.as_str()) {
                continue;
            }
            return parsed
                .get("value")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| {
                    SdrConnectError::Protocol(format!("get_property_response for {property:?} had no value"))
                });
        }
    }

    pub fn get_u64(&self, property: &str) -> Result<u64, SdrConnectError> {
        self.get_property(property)?
            .trim()
            .parse()
            .map_err(|e| SdrConnectError::Protocol(format!("{property} was not a u64: {e}")))
    }

    pub fn get_f64(&self, property: &str) -> Result<f64, SdrConnectError> {
        self.get_property(property)?
            .trim()
            .parse()
            .map_err(|e| SdrConnectError::Protocol(format!("{property} was not an f64: {e}")))
    }

    pub fn get_i32(&self, property: &str) -> Result<i32, SdrConnectError> {
        self.get_property(property)?
            .trim()
            .parse()
            .map_err(|e| SdrConnectError::Protocol(format!("{property} was not an i32: {e}")))
    }

    pub fn get_bool(&self, property: &str) -> Result<bool, SdrConnectError> {
        let v = self.get_property(property)?;
        match v.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            other => Err(SdrConnectError::Protocol(format!(
                "{property} was not a bool: {other:?}"
            ))),
        }
    }

    pub fn set_vfo_frequency_hz(&self, hz: u64) -> Result<(), SdrConnectError> {
        self.set_property(property::VFO_FREQUENCY, hz.to_string())
    }
    pub fn vfo_frequency_hz(&self) -> Result<u64, SdrConnectError> {
        self.get_u64(property::VFO_FREQUENCY)
    }

    pub fn set_center_frequency_hz(&self, hz: u64) -> Result<(), SdrConnectError> {
        self.set_property(property::CENTER_FREQUENCY, hz.to_string())
    }
    pub fn center_frequency_hz(&self) -> Result<u64, SdrConnectError> {
        self.get_u64(property::CENTER_FREQUENCY)
    }

    pub fn set_filter_bandwidth_hz(&self, hz: u32) -> Result<(), SdrConnectError> {
        self.set_property(property::FILTER_BANDWIDTH, hz.to_string())
    }
    pub fn filter_bandwidth_hz(&self) -> Result<u32, SdrConnectError> {
        self.get_u64(property::FILTER_BANDWIDTH).map(|v| v as u32)
    }

    /// SDRconnect's demodulator name as IT spells it (its own vocabulary, not rigctld's — see
    /// `sdrconnect_daemon`'s mode-mapping functions for the translation either way).
    pub fn set_demodulator(&self, name: &str) -> Result<(), SdrConnectError> {
        self.set_property(property::DEMODULATOR, name)
    }
    pub fn demodulator(&self) -> Result<String, SdrConnectError> {
        self.get_property(property::DEMODULATOR)
    }

    pub fn set_lna_state(&self, state: i32) -> Result<(), SdrConnectError> {
        self.set_property(property::LNA_STATE, state.to_string())
    }
    pub fn lna_state(&self) -> Result<i32, SdrConnectError> {
        self.get_i32(property::LNA_STATE)
    }
    pub fn lna_state_min(&self) -> Result<i32, SdrConnectError> {
        self.get_i32(property::LNA_STATE_MIN)
    }
    pub fn lna_state_max(&self) -> Result<i32, SdrConnectError> {
        self.get_i32(property::LNA_STATE_MAX)
    }

    pub fn set_agc_enable(&self, on: bool) -> Result<(), SdrConnectError> {
        self.set_property(property::AGC_ENABLE, if on { "true" } else { "false" })
    }
    pub fn agc_enable(&self) -> Result<bool, SdrConnectError> {
        self.get_bool(property::AGC_ENABLE)
    }

    pub fn sample_rate_hz(&self) -> Result<f64, SdrConnectError> {
        self.get_f64(property::SAMPLE_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_name_the_property_and_do_not_panic() {
        let e = SdrConnectError::Timeout("no response for property \"lna_state\" within 5s".into());
        assert!(e.to_string().contains("lna_state"));
    }
}

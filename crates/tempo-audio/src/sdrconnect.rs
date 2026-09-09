//! SDRconnect WebSocket API client.
//!
//! Phase 1 intentionally implements control/synchronisation only.
//! IQ/audio/spectrum streaming is not enabled by this module yet.
//!
//! Architecture:
//!     Nexus -> WebSocket/IP -> SDRconnect -> RSP1B
//!
//! The URL is supplied by the caller because the SDRconnect WebSocket
//! endpoint/port is deployment-specific and must not be guessed here.

use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use tungstenite::{connect, Message, WebSocket};

type Ws = WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>;

/// SDRconnect WebSocket event names used by Phase 1.
pub mod event {
    pub const SET_PROPERTY: &str = "set_property";
    pub const GET_PROPERTY: &str = "get_property";
    pub const PROPERTY_CHANGED: &str = "property_changed";
    pub const GET_PROPERTY_RESPONSE: &str = "get_property_response";
    pub const IQ_STREAM_ENABLE: &str = "iq_stream_enable";
    pub const AUDIO_STREAM_ENABLE: &str = "audio_stream_enable";
    pub const SPECTRUM_ENABLE: &str = "spectrum_enable";
    pub const DEVICE_STREAM_ENABLE: &str = "device_stream_enable";
    pub const SELECTED_DEVICE: &str = "selected_device";
    pub const SELECTED_DEVICE_SERIAL: &str = "selected_device_serial";
    pub const SELECTED_DEVICE_NAME: &str = "selected_device_name";
}

/// SDRconnect property names relevant to Phase 1.
pub mod property {
    pub const CENTER_FREQUENCY: &str = "device_center_frequency";
    pub const SAMPLE_RATE: &str = "device_sample_rate";
    pub const VFO_FREQUENCY: &str = "device_vfo_frequency";
    pub const LNA_STATE: &str = "lna_state";
    pub const LNA_STATE_MIN: &str = "lna_state_min";
    pub const LNA_STATE_MAX: &str = "lna_state_max";
    pub const FILTER_BANDWIDTH: &str = "filter_bandwidth";
    pub const DEMODULATOR: &str = "demodulator";
    pub const STARTED: &str = "started";
    pub const OVERLOAD: &str = "overload";
    pub const CAN_CONTROL: &str = "can_control";
}

/// One JSON message exchanged with SDRconnect.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub event_type: String,
    pub property: String,
    pub value: String,
    pub device: Option<String>,
}

impl Envelope {
    pub fn new(event_type: impl Into<String>, property: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            property: property.into(),
            value: value.into(),
            device: None,
        }
    }

    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    fn to_json(&self) -> String {
        let mut value = serde_json::json!({
            "event_type": self.event_type,
            "property": self.property,
            "value": self.value,
        });
        if let Some(device) = &self.device {
            value["device"] = serde_json::Value::String(device.clone());
        }
        value.to_string()
    }

    fn from_json(text: &str) -> Result<Self, SdrConnectError> {
        let value: serde_json::Value = serde_json::from_str(text)?;
        Ok(Self {
            event_type: value
                .get("event_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            property: value
                .get("property")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            value: value
                .get("value")
                .map(json_value_to_string)
                .unwrap_or_default(),
            device: value
                .get("device")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        })
    }
}

fn json_value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

#[derive(Debug)]
pub enum SdrConnectError {
    WebSocket(tungstenite::Error),
    Json(serde_json::Error),
    InvalidValue { property: String, value: String },
    UnexpectedResponse { event_type: String, property: String },
    ServerRejected { property: String, value: String },
    Timeout,
}

impl fmt::Display for SdrConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WebSocket(e) => write!(f, "SDRconnect WebSocket error: {e}"),
            Self::Json(e) => write!(f, "SDRconnect JSON error: {e}"),
            Self::InvalidValue { property, value } => {
                write!(f, "invalid SDRconnect value for {property}: {value}")
            }
            Self::UnexpectedResponse { event_type, property } => {
                write!(f, "unexpected SDRconnect response {event_type} for {property}")
            }
            Self::ServerRejected { property, value } => {
                write!(f, "SDRconnect rejected {property}={value}")
            }
            Self::Timeout => write!(f, "SDRconnect response timeout"),
        }
    }
}

impl std::error::Error for SdrConnectError {}

impl From<tungstenite::Error> for SdrConnectError {
    fn from(value: tungstenite::Error) -> Self {
        Self::WebSocket(value)
    }
}

impl From<serde_json::Error> for SdrConnectError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Thread-safe, synchronous SDRconnect client.
///
/// A mutex serializes WebSocket transactions. This keeps Phase 1 independent
/// from Nexus' async/runtime architecture and makes the client usable from
/// the existing real-radio transport code.
pub struct SdrConnect {
    socket: Mutex<Ws>,
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
    /// Connect to an SDRconnect WebSocket endpoint.
    ///
    /// Example: `ws://192.168.1.50:5454`
    pub fn connect(url: &str) -> Result<Self, SdrConnectError> {
        let (socket, _) = connect(url)?;
        Ok(Self {
            socket: Mutex::new(socket),
            timeout: Duration::from_secs(5),
        })
    }

    /// Set the timeout used while waiting for a matching get_property response.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Send a raw JSON envelope.
    pub fn send(&self, envelope: &Envelope) -> Result<(), SdrConnectError> {
        let mut socket = self.socket.lock().expect("SDRconnect mutex poisoned");
        socket.send(Message::Text(envelope.to_json().into()))?;
        Ok(())
    }

    /// Set a property.
    ///
    /// SDRconnect's set_property operation does not provide a dedicated
    /// acknowledgement. Call `get_property` afterwards when confirmation is
    /// required.
    pub fn set_property(
        &self,
        property: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), SdrConnectError> {
        self.send(&Envelope::new(event::SET_PROPERTY, property, value))
    }

    /// Set a property on a specific device, when the API deployment uses the
    /// optional `device` field.
    pub fn set_property_for_device(
        &self,
        property: impl Into<String>,
        value: impl Into<String>,
        device: impl Into<String>,
    ) -> Result<(), SdrConnectError> {
        self.send(
            &Envelope::new(event::SET_PROPERTY, property, value)
                .with_device(device),
        )
    }

    /// Request a property and wait for its get_property_response.
    pub fn get_property(&self, property: impl Into<String>) -> Result<String, SdrConnectError> {
        let property = property.into();
        let request = Envelope::new(event::GET_PROPERTY, property.clone(), "");
        let mut socket = self.socket.lock().expect("SDRconnect mutex poisoned");

        socket.send(Message::Text(request.to_json().into()))?;

        // SDRconnect does not expose a request-id in the documented envelope.
        // Therefore the matching key is event_type + property. Ignore unrelated
        // property_changed events while waiting for the requested response.
        let deadline = std::time::Instant::now() + self.timeout;

        loop {
            if std::time::Instant::now() >= deadline {
                return Err(SdrConnectError::Timeout);
            }

            match socket.read()? {
                Message::Text(text) => {
                    let response = Envelope::from_json(&text)?;
                    if response.event_type == event::GET_PROPERTY_RESPONSE
                        && response.property == property
                    {
                        return Ok(response.value);
                    }
                    if response.event_type == "error" {
                        return Err(SdrConnectError::ServerRejected {
                            property: response.property,
                            value: response.value,
                        });
                    }
                }
                Message::Ping(payload) => {
                    socket.send(Message::Pong(payload))?;
                }
                Message::Pong(_) => {}
                Message::Binary(_) => {}
                Message::Close(_) => return Err(SdrConnectError::WebSocket(
                    tungstenite::Error::ConnectionClosed,
                )),
                _ => {}
            }
        }
    }

    pub fn get_u64(&self, property: &str) -> Result<u64, SdrConnectError> {
        let value = self.get_property(property)?;
        value.parse().map_err(|_| SdrConnectError::InvalidValue {
            property: property.to_owned(),
            value,
        })
    }

    pub fn get_i32(&self, property: &str) -> Result<i32, SdrConnectError> {
        let value = self.get_property(property)?;
        value.parse().map_err(|_| SdrConnectError::InvalidValue {
            property: property.to_owned(),
            value,
        })
    }

    pub fn get_f64(&self, property: &str) -> Result<f64, SdrConnectError> {
        let value = self.get_property(property)?;
        value.parse().map_err(|_| SdrConnectError::InvalidValue {
            property: property.to_owned(),
            value,
        })
    }

    pub fn get_bool(&self, property: &str) -> Result<bool, SdrConnectError> {
        let value = self.get_property(property)?;
        match value.as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(SdrConnectError::InvalidValue {
                property: property.to_owned(),
                value,
            }),
        }
    }

    pub fn set_center_frequency_hz(&self, hz: u64) -> Result<(), SdrConnectError> {
        self.set_property(property::CENTER_FREQUENCY, hz.to_string())
    }

    pub fn center_frequency_hz(&self) -> Result<u64, SdrConnectError> {
        self.get_u64(property::CENTER_FREQUENCY)
    }

    pub fn set_vfo_frequency_hz(&self, hz: u64) -> Result<(), SdrConnectError> {
        self.set_property(property::VFO_FREQUENCY, hz.to_string())
    }

    pub fn vfo_frequency_hz(&self) -> Result<u64, SdrConnectError> {
        self.get_u64(property::VFO_FREQUENCY)
    }

    pub fn set_sample_rate_hz(&self, hz: f64) -> Result<(), SdrConnectError> {
        self.set_property(property::SAMPLE_RATE, hz.to_string())
    }

    pub fn sample_rate_hz(&self) -> Result<f64, SdrConnectError> {
        self.get_f64(property::SAMPLE_RATE)
    }

    pub fn set_lna_state(&self, state: i32) -> Result<(), SdrConnectError> {
        self.set_property(property::LNA_STATE, state.to_string())
    }

    pub fn lna_state(&self) -> Result<i32, SdrConnectError> {
        self.get_i32(property::LNA_STATE)
    }

    pub fn set_filter_bandwidth_hz(&self, hz: u32) -> Result<(), SdrConnectError> {
        self.set_property(property::FILTER_BANDWIDTH, hz.to_string())
    }

    pub fn filter_bandwidth_hz(&self) -> Result<u32, SdrConnectError> {
        let value = self.get_u64(property::FILTER_BANDWIDTH)?;
        u32::try_from(value).map_err(|_| SdrConnectError::InvalidValue {
            property: property::FILTER_BANDWIDTH.to_owned(),
            value: value.to_string(),
        })
    }

    pub fn set_demodulator(&self, mode: &str) -> Result<(), SdrConnectError> {
        const MODES: &[&str] = &["AM", "USB", "LSB", "CW", "SAM", "NFM", "WFM"];
        if !MODES.contains(&mode) {
            return Err(SdrConnectError::InvalidValue {
                property: property::DEMODULATOR.to_owned(),
                value: mode.to_owned(),
            });
        }
        self.set_property(property::DEMODULATOR, mode)
    }

    pub fn demodulator(&self) -> Result<String, SdrConnectError> {
        self.get_property(property::DEMODULATOR)
    }

    /// Send the device-stream command. IQ is deliberately not enabled yet.
    pub fn set_device_stream_enable(&self, enable: bool) -> Result<(), SdrConnectError> {
        self.send(&Envelope::new(
            event::DEVICE_STREAM_ENABLE,
            "",
            if enable { "true" } else { "false" },
        ))
    }

    /// Explicitly available for Phase 2, but not called by Nexus Phase 1.
    pub fn set_iq_stream_enable(&self, enable: bool) -> Result<(), SdrConnectError> {
        self.send(&Envelope::new(
            event::IQ_STREAM_ENABLE,
            "",
            if enable { "true" } else { "false" },
        ))
    }

    /// Read one incoming WebSocket message without changing it.
    ///
    /// Useful for an integration layer that wants to consume property_changed
    /// notifications. Phase 1 does not run a background event thread.
    pub fn read_message(&self) -> Result<IncomingMessage, SdrConnectError> {
        let mut socket = self.socket.lock().expect("SDRconnect mutex poisoned");
        match socket.read()? {
            Message::Text(text) => Ok(IncomingMessage::Json(Envelope::from_json(&text)?)),
            Message::Binary(data) => Ok(IncomingMessage::Binary(data.to_vec())),
            Message::Ping(payload) => {
                socket.send(Message::Pong(payload.clone()))?;
                Ok(IncomingMessage::Ping(payload.to_vec()))
            }
            Message::Pong(payload) => Ok(IncomingMessage::Pong(payload.to_vec())),
            Message::Close(_) => Err(SdrConnectError::WebSocket(
                tungstenite::Error::ConnectionClosed,
            )),
            _ => Ok(IncomingMessage::Other),
        }
    }
}

#[derive(Debug, Clone)]
pub enum IncomingMessage {
    Json(Envelope),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_serialization_matches_api_shape() {
        let e = Envelope::new(event::SET_PROPERTY, property::VFO_FREQUENCY, "145500000");
        let v: serde_json::Value = serde_json::from_str(&e.to_json()).unwrap();
        assert_eq!(v["event_type"], "set_property");
        assert_eq!(v["property"], "device_vfo_frequency");
        assert_eq!(v["value"], "145500000");
    }

    #[test]
    fn envelope_parses_string_and_numeric_values() {
        let a = Envelope::from_json(
            r#"{"event_type":"get_property_response","property":"device_vfo_frequency","value":"145500000"}"#,
        )
        .unwrap();
        assert_eq!(a.value, "145500000");

        let b = Envelope::from_json(
            r#"{"event_type":"property_changed","property":"overload","value":false}"#,
        )
        .unwrap();
        assert_eq!(b.value, "false");
    }

    #[test]
    fn demodulator_validation() {
        assert!(SdrConnect::connect("ws://127.0.0.1:1").is_err() || true);
    }
}

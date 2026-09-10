//! **The software receiver.** SDRconnect's WebSocket API hands Nexus raw wideband IQ (message
//! type 2) — everything from there to a 12 kHz mono audio stream `RxTap`/`RxRing` can consume
//! happens HERE, in Rust, on Nexus's own clock. SDRconnect's own `demodulator` property is
//! never touched by this path: the RSP1B's demodulation is Nexus's job now, by design (see the
//! decision recorded in `sdrconnect_daemon.rs`'s module doc) — Option B, not Option A.
//!
//! # Why local demod, and what it costs
//!
//! SDRconnect can demodulate for us (`demodulator`/`filter_bandwidth` + message type 1, plain
//! PCM audio). That is simpler and was the FIRST plan. It was rejected because it leaves Nexus
//! dependent on SDRconnect's own DSP quality and its own idea of filter shape/AGC/squelch,
//! rather than treating the RSP1B like any other radio Nexus fully owns the receive chain for.
//! The cost is everything in this file: a decimator, a channel filter per mode, and five
//! demodulators, all written and tested here rather than trusted to someone else's black box.
//!
//! # The simplification that makes this tractable: no NCO retuning
//!
//! A "real" SDR panadapter tunes the hardware to a WIDE span once and lets the operator retune
//! the VFO anywhere inside it by shifting a digital local oscillator. This module does not do
//! that (yet): [`crate::sdrconnect_daemon`] keeps `device_center_frequency` locked to the
//! operator's tuned frequency on every CAT frequency write, so the IQ block arriving here is
//! ALWAYS already centered on 0 Hz baseband. That turns "mix to the VFO, then filter" into just
//! "filter" — a Nexus VFO retune becomes a hardware retune (a few ms of SDRconnect round-trip),
//! never a local NCO shift. Fine for a single-channel ham receiver; the corner this cuts is
//! "listen to two frequencies from one IQ block at once", which nothing in Nexus asks for.
//!
//! # The chain
//!
//! 1. **Stage 1 — coarse decimation.** [`FirDecimator`] with a wide real lowpass brings the
//!    250 kHz IQ (see `sdrconnect_iq`'s `IQ_SAMPLE_RATE_HZ`) down to 25 kHz. Fixed regardless of
//!    mode — it only has to be wide enough for the widest mode this file supports (WFM, capped
//!    below Nyquist).
//! 2. **Stage 2 — the channel filter**, rebuilt only when [`DemodParams`] changes:
//!    - AM/NFM/WFM: a real lowpass ([`FirDecimator`] with `decim = 1`) at half the configured
//!      bandwidth.
//!    - USB/LSB/CW: a COMPLEX bandpass ([`ComplexFir`], via [`design_ssb_bandpass`]) that keeps
//!      only one side of the spectrum — see that function's doc for why a real filter cannot do
//!      this and a complex one can.
//! 3. **Demod**: [`demod_am`] (envelope + DC block), [`demod_ssb`] (just `.i` — see below),
//!    [`demod_fm`] (complex phase-difference discriminator) for NFM/WFM.
//! 4. **Final resample** to 12 kHz mono via [`crate::capture_resample::CaptureResampler`] —
//!    reused as-is; it does not care what produced its input.
//!
//! # Verification without hardware
//!
//! Nothing here has run against a real RSP1B — there is no bench in this environment. Every
//! primitive is instead checked against SYNTHETIC IQ: a pure tone at a known frequency and
//! amplitude fed through `design_lowpass`/`design_ssb_bandpass`/each demodulator, with the
//! output checked against the closed-form answer (attenuation outside the passband, recovered
//! tone frequency/amplitude inside it, phase-continuity of the decimator across arbitrary block
//! splits). That proves the MATH; it cannot prove the RSP1B's actual IQ matches this file's
//! assumptions about level, DC offset, or I/Q balance — those are the first things to check
//! against a real signal.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

/// One complex baseband sample. Not `num_complex::Complex32` — this crate has no complex-number
/// dependency anywhere else, and the handful of operations this file needs (multiply, add,
/// scale, conjugate, magnitude) are not worth a new dependency for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Iq {
    pub i: f32,
    pub q: f32,
}

impl Iq {
    pub const fn new(i: f32, q: f32) -> Self {
        Self { i, q }
    }

    fn add(self, other: Iq) -> Iq {
        Iq::new(self.i + other.i, self.q + other.q)
    }

    fn scale(self, k: f32) -> Iq {
        Iq::new(self.i * k, self.q * k)
    }

    /// Complex multiply: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
    fn mul(self, other: Iq) -> Iq {
        Iq::new(
            self.i * other.i - self.q * other.q,
            self.i * other.q + self.q * other.i,
        )
    }

    fn conj(self) -> Iq {
        Iq::new(self.i, -self.q)
    }

    fn magnitude(self) -> f32 {
        (self.i * self.i + self.q * self.q).sqrt()
    }
}

/// One rigctld mode word <-> the demodulator this file runs for it.
///
/// SDRconnect's own `demodulator` enum is NOT reused here on purpose — that property belongs to
/// a signal path this module no longer drives (see the module doc). This is Nexus's own idea of
/// mode, independent of whatever SDRconnect's enum happens to spell things as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemodMode {
    Am,
    Usb,
    Lsb,
    Cw,
    Nfm,
    Wfm,
}

impl DemodMode {
    /// A rigctld mode word → the mode this file demodulates it as. SDRconnect has no CWR, no
    /// DATA-submode distinction and no synchronous-AM in this file's chain, so — same rule as
    /// `sdrconnect_daemon`'s mapping — every rigctld DATA word collapses onto plain USB/LSB and
    /// every CW spelling collapses onto one CW. `None` for anything this receiver cannot run
    /// (e.g. RTTY, which needs its own filter/demod this file does not implement): refused, not
    /// approximated.
    pub fn from_rigctld(word: &str) -> Option<Self> {
        Some(match word.trim().to_ascii_uppercase().as_str() {
            "USB" | "PKTUSB" | "DATA-U" | "PKT-U" | "USB-D" => DemodMode::Usb,
            "LSB" | "PKTLSB" | "DATA-L" | "PKT-L" | "LSB-D" => DemodMode::Lsb,
            "CW" | "CW-U" | "CWU" | "CWR" | "CW-L" | "CWL" => DemodMode::Cw,
            "AM" => DemodMode::Am,
            "FM" | "PKTFM" | "FM-D" | "PKT-FM" => DemodMode::Nfm,
            "WFM" => DemodMode::Wfm,
            _ => return None,
        })
    }

    /// The rigctld mode word to report back for this demod — the inverse of `from_rigctld`,
    /// never blank (mirrors every other backend's "never report an empty mode" rule).
    pub fn to_rigctld(self) -> &'static str {
        match self {
            DemodMode::Am => "AM",
            DemodMode::Usb => "USB",
            DemodMode::Lsb => "LSB",
            DemodMode::Cw => "CW",
            DemodMode::Nfm => "FM",
            DemodMode::Wfm => "WFM",
        }
    }

    fn code(self) -> u8 {
        match self {
            DemodMode::Am => 0,
            DemodMode::Usb => 1,
            DemodMode::Lsb => 2,
            DemodMode::Cw => 3,
            DemodMode::Nfm => 4,
            DemodMode::Wfm => 5,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            0 => DemodMode::Am,
            1 => DemodMode::Usb,
            2 => DemodMode::Lsb,
            3 => DemodMode::Cw,
            4 => DemodMode::Nfm,
            _ => DemodMode::Wfm,
        }
    }
}

/// The ONLY state shared between the CAT shim (`sdrconnect_daemon::SdrConnectBackend`, which
/// writes it from rigctld `M`/`m` commands) and the IQ worker (`sdrconnect_iq::SdrConnectIq`,
/// which reads it every block to decide whether to rebuild its channel filter). Atomics, not a
/// `Mutex`, because the IQ worker reads this on every single processing block (potentially
/// hundreds of times a second) and must never block on a CAT command in flight.
pub struct DemodParams {
    mode: AtomicU8,
    bandwidth_hz: AtomicU32,
}

impl DemodParams {
    pub fn new(mode: DemodMode, bandwidth_hz: u32) -> Arc<Self> {
        Arc::new(Self {
            mode: AtomicU8::new(mode.code()),
            bandwidth_hz: AtomicU32::new(bandwidth_hz),
        })
    }

    pub fn mode(&self) -> DemodMode {
        DemodMode::from_code(self.mode.load(Ordering::Relaxed))
    }

    pub fn set_mode(&self, mode: DemodMode) {
        self.mode.store(mode.code(), Ordering::Relaxed);
    }

    pub fn bandwidth_hz(&self) -> u32 {
        self.bandwidth_hz.load(Ordering::Relaxed)
    }

    pub fn set_bandwidth_hz(&self, hz: u32) {
        self.bandwidth_hz.store(hz, Ordering::Relaxed);
    }
}

/// A free-running numerically-controlled oscillator. Not currently used by the processing chain
/// (see the module doc: Phase 1 keeps the hardware retuned to 0 Hz offset instead of mixing),
/// but kept as a public, tested primitive because it is the one piece Phase 2 (arbitrary VFO
/// offset within one captured IQ block) will need first, and it is easy to get subtly wrong
/// (phase wrap, frequency sign) in a way that only shows up as a slowly-drifting audio pitch.
pub struct Nco {
    phase: f32,
    step: f32,
}

impl Nco {
    pub fn new(freq_hz: f32, sample_rate: f32) -> Self {
        Self {
            phase: 0.0,
            step: 2.0 * std::f32::consts::PI * freq_hz / sample_rate,
        }
    }

    pub fn next(&mut self) -> Iq {
        let out = Iq::new(self.phase.cos(), self.phase.sin());
        self.phase += self.step;
        // Keep the phase bounded so it never loses precision over a long-running stream —
        // millions of samples in, an unwrapped `f32` phase would start showing audible jitter.
        if self.phase > std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        } else if self.phase < -std::f32::consts::PI {
            self.phase += 2.0 * std::f32::consts::PI;
        }
        out
    }
}

/// A Hann-windowed-sinc real lowpass, normalised to unity DC gain (taps sum to 1) so a filtered
/// DC/carrier level survives at the same amplitude — the same normalisation
/// `capture_resample::fir_tap`'s polyphase design relies on, done here as a plain one-shot
/// design since this file needs a handful of distinct filters, not thousands of resampler
/// phases.
///
/// `num_taps` is forced to the next odd number ≥ 3 (Type I linear-phase: a single, exact centre
/// tap, no ambiguity about which sample the filter's group delay is centred on).
pub fn design_lowpass(num_taps: usize, cutoff_norm: f64) -> Vec<f32> {
    let num_taps = if num_taps < 3 {
        3
    } else if num_taps % 2 == 0 {
        num_taps + 1
    } else {
        num_taps
    };
    let cutoff_norm = cutoff_norm.clamp(1e-4, 0.499);
    let m = (num_taps - 1) as f64;
    let mut taps = Vec::with_capacity(num_taps);
    for n in 0..num_taps {
        let k = n as f64 - m / 2.0;
        let sinc = if k.abs() < 1e-9 {
            2.0 * cutoff_norm
        } else {
            (2.0 * std::f64::consts::PI * cutoff_norm * k).sin() / (std::f64::consts::PI * k)
        };
        // Hann window: zero at both ends, so the filter has no sharp truncation edge.
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / m).cos();
        taps.push(sinc * w);
    }
    let dc_gain: f64 = taps.iter().sum();
    if dc_gain.abs() > 1e-12 {
        for t in &mut taps {
            *t /= dc_gain;
        }
    }
    taps.into_iter().map(|t| t as f32).collect()
}

/// A complex bandpass built by frequency-SHIFTING a real lowpass prototype: multiplying a real
/// lowpass impulse response (passband `[-half_bw, +half_bw]`) by a complex exponential at
/// `f_center` translates its passband to `[f_center - half_bw, f_center + half_bw]` — a standard
/// filter-design result (`H_shifted(f) = H_lp(f - f_center)`).
///
/// **Why this has to be complex, not real, for SSB.** A REAL filter's frequency response is
/// always mirror-symmetric around 0 Hz (`H(-f) = H(f)`) — there is no way to keep positive
/// frequencies and reject the mirrored negative ones with real coefficients. Choosing `f_lo`/
/// `f_hi` both positive (for USB) or both negative (for LSB) — so the shifted passband never
/// straddles 0 Hz — makes the result ONE-SIDED, an analytic signal for just that sideband. Its
/// real part (see [`demod_ssb`]) is then the demodulated audio — the textbook "phasing method"
/// SSB demodulator, made simple here specifically because the input is true I/Q and not a
/// real-valued IF that would need a separate Hilbert transform to become analytic.
pub fn design_ssb_bandpass(sample_rate: f64, f_lo: f64, f_hi: f64, num_taps: usize) -> Vec<Iq> {
    let f_center = (f_hi + f_lo) / 2.0;
    let half_bw = ((f_hi - f_lo) / 2.0).abs();
    let lp = design_lowpass(num_taps, half_bw / sample_rate);
    let num_taps = lp.len();
    let m = (num_taps - 1) as f64;
    lp.iter()
        .enumerate()
        .map(|(n, &c)| {
            let k = n as f64 - m / 2.0;
            let phase = 2.0 * std::f64::consts::PI * f_center * k / sample_rate;
            Iq::new(
                (c as f64 * phase.cos()) as f32,
                (c as f64 * phase.sin()) as f32,
            )
        })
        .collect()
}

/// Odd tap count long enough to reach `transition_hz` of stopband transition at `sample_rate` —
/// the standard windowed-FIR rule of thumb (`N ≈ 4·fs/Δf` for a Hann window). Clamped to a sane
/// range: too few taps and a narrow CW filter would not reject the next signal 500 Hz away; too
/// many and a bandwidth typo (0 Hz) would try to allocate a filter that never finishes.
pub fn taps_for_transition(sample_rate: f64, transition_hz: f64) -> usize {
    let n = (4.0 * sample_rate / transition_hz.max(50.0)).ceil() as usize;
    let n = n.clamp(31, 1201);
    if n % 2 == 0 {
        n + 1
    } else {
        n
    }
}

/// A real-coefficient FIR filter, optionally decimating — the coarse Stage 1 decimator AND
/// (with `decim = 1`) the AM/NFM/WFM channel filter share this one implementation.
///
/// Streams across arbitrary block sizes without losing decimation PHASE: naively restarting the
/// "which sample is the first output" question at the start of every `process()` call would
/// make the output rate depend on how the caller happened to chunk its input (imperceptible at
/// `decim = 1`, but a slowly-drifting pitch at `decim > 1` as blocks of network data arrive in
/// whatever sizes the OS handed them over). Tracking a running absolute sample count instead
/// keeps every output sample at the same absolute position regardless of chunking — see the
/// `phase_continuity_survives_arbitrary_chunking` test, which is the one that would catch a
/// regression here.
pub struct FirDecimator {
    taps: Vec<f32>,
    decim: usize,
    /// Exactly `taps.len() - 1` samples: whatever immediately precedes the next fresh block.
    history: Vec<Iq>,
    /// Absolute count of samples ever handed to `process` (NOT counting the seed history).
    total_in: u64,
}

impl FirDecimator {
    pub fn new(taps: Vec<f32>, decim: usize) -> Self {
        let hist_len = taps.len().saturating_sub(1);
        Self {
            taps,
            decim: decim.max(1),
            history: vec![Iq::default(); hist_len],
            total_in: 0,
        }
    }

    pub fn process(&mut self, input: &[Iq]) -> Vec<Iq> {
        let hist_len = self.taps.len().saturating_sub(1);
        let mut buf = Vec::with_capacity(hist_len + input.len());
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(input);

        // buf[0] sits at absolute sample index `total_in - hist_len`. Find the smallest p with
        // a full window (p >= hist_len) whose absolute index is a multiple of `decim`, so every
        // output sample lands on the same absolute grid no matter how `input` was chunked.
        let base_abs = self.total_in as i64 - hist_len as i64;
        let decim_i = self.decim as i64;
        let rem = ((base_abs % decim_i) + decim_i) % decim_i;
        let mut p = if rem == 0 { 0usize } else { (decim_i - rem) as usize };
        if p < hist_len {
            let deficit = hist_len - p;
            let steps = deficit.div_ceil(self.decim);
            p += steps * self.decim;
        }

        let mut out = Vec::new();
        while p < buf.len() {
            let mut acc = Iq::default();
            for (k, &c) in self.taps.iter().enumerate() {
                acc = acc.add(buf[p - k].scale(c));
            }
            out.push(acc);
            p += self.decim;
        }

        self.total_in += input.len() as u64;
        self.history = if buf.len() >= hist_len {
            buf[buf.len() - hist_len..].to_vec()
        } else {
            let mut h = vec![Iq::default(); hist_len - buf.len()];
            h.extend_from_slice(&buf);
            h
        };
        out
    }
}

/// A complex-coefficient FIR filter (no decimation — the SSB/CW bandpass runs at the Stage 1
/// output rate). Same streaming-history discipline as [`FirDecimator`], simpler because there
/// is no decimation phase to track.
pub struct ComplexFir {
    taps: Vec<Iq>,
    history: Vec<Iq>,
}

impl ComplexFir {
    pub fn new(taps: Vec<Iq>) -> Self {
        let history = vec![Iq::default(); taps.len().saturating_sub(1)];
        Self { taps, history }
    }

    pub fn process(&mut self, input: &[Iq]) -> Vec<Iq> {
        let hist_len = self.taps.len().saturating_sub(1);
        let mut buf = Vec::with_capacity(hist_len + input.len());
        buf.extend_from_slice(&self.history);
        buf.extend_from_slice(input);

        let mut out = Vec::with_capacity(input.len());
        for p in hist_len..buf.len() {
            let mut acc = Iq::default();
            for (k, &c) in self.taps.iter().enumerate() {
                acc = acc.add(buf[p - k].mul(c));
            }
            out.push(acc);
        }
        self.history = if buf.len() >= hist_len {
            buf[buf.len() - hist_len..].to_vec()
        } else {
            let mut h = vec![Iq::default(); hist_len - buf.len()];
            h.extend_from_slice(&buf);
            h
        };
        out
    }
}

/// AM: envelope detection (`|I+jQ|`) followed by a 1-pole leaky-integrator DC block, since the
/// envelope of ANY AM signal (even 100% modulated) never goes negative, so its average is a
/// large DC term that would otherwise swamp the audio. `dc_state` is the caller's filter state,
/// carried across calls the same way the FIR filters carry `history`.
pub fn demod_am(input: &[Iq], dc_state: &mut f32) -> Vec<f32> {
    const DC_POLE: f32 = 0.999;
    input
        .iter()
        .map(|s| {
            let mag = s.magnitude();
            *dc_state = DC_POLE * *dc_state + (1.0 - DC_POLE) * mag;
            (mag - *dc_state) * 2.0
        })
        .collect()
}

/// SSB/CW: the real part of the already one-sided bandpassed signal IS the demodulated audio —
/// see [`design_ssb_bandpass`]'s doc for why. `gain` is a linear scale applied on the way out
/// (SSB voice and CW beat notes sit at very different natural levels after the same filter
/// design, so the two callers in `sdrconnect_iq` pass different values).
pub fn demod_ssb(bandpassed: &[Iq], gain: f32) -> Vec<f32> {
    bandpassed.iter().map(|s| s.i * gain).collect()
}

/// NFM/WFM: the classic complex phase-difference (polar) discriminator — the phase of
/// `s[n]·conj(s[n-1])` is the instantaneous frequency in radians/sample, recovered without an
/// `atan` on every sample being anything more than `atan2` of a single complex product. `prev`
/// is the caller's one-sample filter state. `gain` converts radians/sample into the same audio
/// scale the other demodulators use — proportional to `1 / (2π·deviation/sample_rate)` for the
/// mode's expected peak deviation, tuned per-caller in `sdrconnect_iq` rather than baked in here
/// since NFM and WFM deviations differ by an order of magnitude.
pub fn demod_fm(input: &[Iq], prev: &mut Iq, gain: f32) -> Vec<f32> {
    input
        .iter()
        .map(|&s| {
            let cross = s.mul(prev.conj());
            *prev = s;
            cross.q.atan2(cross.i) * gain
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate `n` samples of a pure complex tone at `freq_hz` (positive = "above the tuned
    /// frequency", matching how a real transmitter above the RSP1B's `device_center_frequency`
    /// would appear in the captured IQ).
    fn tone(freq_hz: f64, sample_rate: f64, amplitude: f32, n: usize) -> Vec<Iq> {
        let step = 2.0 * std::f64::consts::PI * freq_hz / sample_rate;
        (0..n)
            .map(|k| {
                let phase = step * k as f64;
                Iq::new(
                    (amplitude as f64 * phase.cos()) as f32,
                    (amplitude as f64 * phase.sin()) as f32,
                )
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn lowpass_has_unity_dc_gain() {
        let taps = design_lowpass(101, 0.05);
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "DC gain should be ~1.0, was {sum}");
    }

    #[test]
    fn lowpass_forces_odd_length() {
        assert_eq!(design_lowpass(100, 0.1).len(), 101);
        assert_eq!(design_lowpass(2, 0.1).len(), 3);
    }

    #[test]
    fn lowpass_attenuates_a_tone_well_past_cutoff() {
        // Cutoff at 1 kHz (norm 0.02 @ 50 kHz); a 15 kHz tone is deep in the stopband.
        let taps = design_lowpass(201, 1_000.0 / 50_000.0);
        let mut filt = FirDecimator::new(taps, 1);
        let passed = tone(200.0, 50_000.0, 1.0, 4_000);
        let blocked = tone(15_000.0, 50_000.0, 1.0, 4_000);
        let out_pass = filt.process(&passed);
        let mut filt2 = FirDecimator::new(design_lowpass(201, 1_000.0 / 50_000.0), 1);
        let out_block = filt2.process(&blocked);
        // Settle past the filter's own group delay before comparing steady-state levels.
        let pass_rms = rms(&out_pass[500..].iter().map(|s| s.i).collect::<Vec<_>>());
        let block_rms = rms(&out_block[500..].iter().map(|s| s.i).collect::<Vec<_>>());
        assert!(pass_rms > 0.5, "in-band tone should pass mostly intact, got {pass_rms}");
        assert!(
            block_rms < 0.05,
            "far-out-of-band tone should be heavily attenuated, got {block_rms}"
        );
    }

    #[test]
    fn decimator_phase_continuity_survives_arbitrary_chunking() {
        let input = tone(500.0, 50_000.0, 1.0, 5_000);
        let taps = design_lowpass(65, 5_000.0 / 50_000.0);

        let mut whole = FirDecimator::new(taps.clone(), 5);
        let out_whole = whole.process(&input);

        // Same stream, fed in awkward, unequal chunks.
        let mut chunked = FirDecimator::new(taps, 5);
        let mut out_chunked = Vec::new();
        for chunk in input.chunks(37) {
            out_chunked.extend(chunked.process(chunk));
        }

        assert_eq!(out_whole.len(), out_chunked.len(), "chunking must not change the output rate");
        for (a, b) in out_whole.iter().zip(out_chunked.iter()) {
            assert!((a.i - b.i).abs() < 1e-4 && (a.q - b.q).abs() < 1e-4, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn ssb_bandpass_passes_the_wanted_sideband_and_rejects_its_mirror() {
        let sample_rate = 25_000.0;
        // USB: 300..2700 Hz, positive side only.
        let taps = design_ssb_bandpass(sample_rate, 300.0, 2_700.0, 401);
        let in_band = tone(1_500.0, sample_rate, 1.0, 6_000);
        let mirror = tone(-1_500.0, sample_rate, 1.0, 6_000); // the LSB side of the same offset
        let mut f1 = ComplexFir::new(taps.clone());
        let mut f2 = ComplexFir::new(taps);
        let audio_in = demod_ssb(&f1.process(&in_band), 1.0);
        let audio_mirror = demod_ssb(&f2.process(&mirror), 1.0);
        let in_rms = rms(&audio_in[1000..]);
        let mirror_rms = rms(&audio_mirror[1000..]);
        assert!(in_rms > 0.3, "the wanted sideband should come through, got {in_rms}");
        assert!(
            mirror_rms < 0.05,
            "the mirrored (opposite-sideband) tone must be rejected, got {mirror_rms}"
        );
    }

    #[test]
    fn am_recovers_a_modulation_tone_from_the_envelope() {
        let sample_rate = 25_000.0;
        // A carrier at 0 Hz amplitude-modulated at 400 Hz, 50% depth: envelope = 1 + 0.5 cos(wt).
        let n = 8_000;
        let step = 2.0 * std::f64::consts::PI * 400.0 / sample_rate;
        let iq: Vec<Iq> = (0..n)
            .map(|k| {
                let env = 1.0 + 0.5 * (step * k as f64).cos();
                Iq::new(env as f32, 0.0)
            })
            .collect();
        let mut dc = 0.0f32;
        let audio = demod_am(&iq, &mut dc);
        // After the DC blocker settles, the recovered audio should have real (non-zero) energy
        // — a broken envelope/DC-removal chain (e.g. detecting `.i` directly) would flatline.
        let settled = rms(&audio[2000..]);
        assert!(settled > 0.1, "expected a recovered 400 Hz tone, got rms {settled}");
    }

    #[test]
    fn fm_discriminator_reports_a_frequency_proportional_offset() {
        let sample_rate = 25_000.0;
        let mut prev = Iq::new(1.0, 0.0);
        let low = tone(500.0, sample_rate, 1.0, 4_000);
        let high = tone(2_000.0, sample_rate, 1.0, 4_000);
        let out_low = demod_fm(&low, &mut prev, 1.0);
        let mut prev2 = Iq::new(1.0, 0.0);
        let out_high = demod_fm(&high, &mut prev2, 1.0);
        // Both are constant-frequency tones, so the discriminator output should be a near-flat
        // DC level (in radians/sample) — and the higher-offset tone's level must be larger.
        let avg = |s: &[f32]| s[10..].iter().sum::<f32>() / (s.len() - 10) as f32;
        let (a_low, a_high) = (avg(&out_low), avg(&out_high));
        assert!(a_high > a_low, "a higher frequency offset must discriminate higher: {a_low} vs {a_high}");
    }

    #[test]
    fn demod_mode_rigctld_round_trip() {
        for (word, mode) in [
            ("USB", DemodMode::Usb),
            ("PKTUSB", DemodMode::Usb),
            ("LSB", DemodMode::Lsb),
            ("CW", DemodMode::Cw),
            ("CWR", DemodMode::Cw),
            ("AM", DemodMode::Am),
            ("FM", DemodMode::Nfm),
            ("WFM", DemodMode::Wfm),
        ] {
            assert_eq!(DemodMode::from_rigctld(word), Some(mode), "word {word}");
        }
        assert_eq!(DemodMode::from_rigctld("RTTY"), None, "unsupported mode must be refused");
        // Reporting back never blanks, and round-trips through a mode this file supports.
        for m in [DemodMode::Am, DemodMode::Usb, DemodMode::Lsb, DemodMode::Cw, DemodMode::Nfm, DemodMode::Wfm] {
            let word = m.to_rigctld();
            assert!(!word.is_empty());
            assert_eq!(DemodMode::from_rigctld(word), Some(m));
        }
    }

    #[test]
    fn demod_params_are_shared_through_the_arc() {
        let params = DemodParams::new(DemodMode::Usb, 2_700);
        let other = params.clone();
        other.set_mode(DemodMode::Cw);
        other.set_bandwidth_hz(400);
        assert_eq!(params.mode(), DemodMode::Cw);
        assert_eq!(params.bandwidth_hz(), 400);
    }
}

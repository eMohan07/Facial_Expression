// Cognitive / behavioral states layer.
// Distinct from the 8 AffectNet emotions HSEmotion is trained on. These are
// secondary states (tired, focused, bored, stressed, engaged, confused, calm)
// derived from baseline-subtracted blendshapes plus literature-grounded
// temporal features (PERCLOS, blink rate, expression entropy).
//
// Numerical anchors (used as fallbacks and threshold offsets):
//   PERCLOS-P80 alarm threshold 0.15 over a 60 s window.
//     Wierwille et al. 1994 (NHTSA Vehicle Behavior Driver Drowsiness study);
//     Dinges & Grace 1998, FHWA Tech Brief TB98-006;
//     Abe 2023, SLEEP Advances zpad006 (modern review).
//   Adult resting blink rate ~ 0.25 Hz (15 blinks/min) measured over ≥60 s.
//     Stern, Walrath & Goldstein 1984, Psychophysiology 21:22-33.
//     Blink rate decreases during visual attention (Stern 1984).
//     Blink rate increases under stress / cognitive load
//     (Doughty 2001; Maffei & Angrilli 2018, Neurosci Lett 736).
//   Confusion AU pattern AU4 + AU7 ("brow lowerer" + "lid tightener").
//     D'Mello & Graesser 2010, Cog & Emotion 24(1):67-76.
//   Boredom signature: expressive flatness + low intensity + slow blink.
//     D'Mello & Graesser 2010; Craig, D'Mello, Witherspoon & Graesser 2008.
//   Stress AU pattern: fear-blend brow (AU1+AU2+AU4) + AU20 + AU24.
//     Harrigan & O'Connell 1996, Pers Indiv Diff 21:205-212;
//     Giannakakis et al. 2017, Biomed Sig Proc & Control 31:89-101.
//   Engagement features: positive V + sustained gaze + AU12/AU1+2 spikes.
//     Whitehill et al. 2014, IEEE T-Affective Comp 5(1):86-98.
//   Calm (Russell circumplex low arousal): Russell 1980 JPSP 39(6).
//   Affect lability (V-flux): Kuppens et al. 2010, 2013, Emotion 10/13.
//
// Per a research pass: no off-the-shelf, in-browser-deployable, permissively-
// licensed classifier exists for these states (LibreFace = USC research only;
// OpenFace = non-commercial; HSEmotion's engagement head was never released
// as a downloadable artifact). So the recipes below are FACS-based, citation-
// anchored. Each state's formula has comments naming the source.
//
// "Excited" is intentionally NOT a separate state -- per Russell circumplex
// and Reisenzein 1994 it's intensified happy, no distinct AU signature.
// "Embarrassed" / "proud" require head-pose pitch which we don't compute yet
// (Keltner 1995, Tracy & Robins 2007 both rely on it).

// Minimum samples needed for state features to be reliable. PERCLOS is
// defined over 60s windows in the literature; Stern 1984 measured blink
// rate over ≥60s. Below this threshold the features are too noisy.
// At ~10 Hz inference, 600 samples ≈ 60s.
export const STATES_MIN_SAMPLES = 600;
// Fallback baseline blink rate when no per-user estimate is available.
// 0.25 Hz = 15 blinks/min (Stern 1984 adult resting median).
const FALLBACK_BLINK_RATE_HZ = 0.25;

export const STATES = [
  "engaged",
  "focused",
  "tired",
  "bored",
  "stressed",
  "confused",
  "calm",
] as const;

export type CognitiveState = (typeof STATES)[number];

// One sample per inference (~10 Hz). 60-sample window ≈ 6 s, 600-sample ≈ 60 s.
export interface LongWindowSample {
  eyeBlink: number;     // raw, NOT baseline-subtracted (PERCLOS uses absolute closure)
  intensity: number;    // L2 of baseline-subtracted blendshape delta vector
  valence: number;      // -1..1
  arousal: number;      // -1..1
}

// Per-user resting metrics, captured during the 3 s baseline calibration.
// Used to convert literature-grounded absolute thresholds into deviations
// from this specific user's "normal" -- so a person with naturally high
// resting blink rate (~0.5 Hz) doesn't always read as stressed, and a
// naturally still person doesn't always read as focused.
export interface UserBaseline {
  blinkRate: number;          // blinks/sec at rest
  intensityVariance: number;  // resting expression flatness
  perclos: number;            // resting fraction-eyes-closed (typically near 0)
  valence: number;            // resting V (often slightly +/- 0)
  arousal: number;            // resting A (often slightly negative for relaxed users)
  valenceFlux: number;        // resting V oscillation magnitude
}

export interface StateInputs {
  // Baseline-subtracted blendshapes for the current frame.
  blendshapes: Record<string, number>;
  // Raw (NOT baseline-subtracted) blendshapes for the current frame; needed
  // for absolute PERCLOS thresholds.
  rawBlendshapes: Record<string, number>;
  valence: number;
  arousal: number;
  intensityNorm: number;
  ambiguous: boolean;
  // Long temporal window (~60 s recommended) for PERCLOS, blink rate, entropy.
  longWindow: ReadonlyArray<LongWindowSample>;
  // Estimate of inference Hz (samples per second). Used to convert blink
  // counts into a per-second rate. Caller should pass ~10 if unsure.
  samplesPerSec: number;
  // The user's resting metrics. When absent, the state formulas fall back
  // to neutral literature-only thresholds (less personalized but still
  // grounded). Highly recommended to provide.
  userBaseline?: UserBaseline;
}

const get = (bs: Record<string, number>, k: string): number => bs[k] ?? 0;
const avg2 = (a: number, b: number): number => (a + b) * 0.5;
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

// ---- Temporal feature extractors (literature-grounded thresholds) ----

// PERCLOS: percentage of time over a window where the eyes are ≥80% closed.
// Threshold ≥ 0.15 over a 60s window → drowsy. Origin: Wierwille et al. 1994
// (Vehicle Behavior Driver Drowsiness study, NHTSA). FHWA TB98-006 Dinges &
// Grace 1998 confirmed 15% as the operational alarm threshold. Modern review:
// Abe 2023, Sleep Advances zpad006.
const PERCLOS_CLOSED_THRESHOLD = 0.8;
function perclos(window: ReadonlyArray<LongWindowSample>): number {
  if (window.length === 0) return 0;
  let closed = 0;
  for (const s of window) if (s.eyeBlink >= PERCLOS_CLOSED_THRESHOLD) closed++;
  return closed / window.length;
}

// Spontaneous blink rate (blinks per second). Detects rising-edge blink
// events (crossing 0.4 going upward). Adult resting baseline is ~12-15
// blinks/min ≈ 0.2-0.25 Hz (Stern, Walrath & Goldstein 1984, Psychophysiology
// 21:22-33). Cognitive/visual-attention literature: rate decreases during
// sustained visual attention (Stern 1984), increases under stress (Doughty
// 2001) and auditory cognitive load (Maffei & Angrilli 2018, Neurosci Lett
// 736:135-139). Direction is task-dependent; we surface rate as a feature
// and let multiple states consume it differently.
const BLINK_EDGE_THRESHOLD = 0.4;
function blinkRatePerSec(
  window: ReadonlyArray<LongWindowSample>,
  samplesPerSec: number,
): number {
  if (window.length < 3) return 0;
  let edges = 0;
  for (let i = 1; i < window.length; i++) {
    if (
      window[i - 1]!.eyeBlink < BLINK_EDGE_THRESHOLD &&
      window[i]!.eyeBlink >= BLINK_EDGE_THRESHOLD
    ) edges++;
  }
  const durSec = window.length / Math.max(1, samplesPerSec);
  return edges / Math.max(1, durSec);
}

// Expression entropy proxy: variance of the intensity series. Used as a
// "flatness" indicator for boredom and as a "sustained" indicator for focus.
// D'Mello 2013 review (J Educ Psych) treats expressive flatness as a
// boredom marker; the variance proxy is the simplest faithful version that
// doesn't need explicit AU clustering.
function intensityVariance(window: ReadonlyArray<LongWindowSample>): number {
  if (window.length < 2) return 0;
  let mean = 0;
  for (const s of window) mean += s.intensity;
  mean /= window.length;
  let acc = 0;
  for (const s of window) acc += (s.intensity - mean) * (s.intensity - mean);
  return acc / window.length;
}

function arousalMean(window: ReadonlyArray<LongWindowSample>): number {
  if (window.length === 0) return 0;
  let m = 0;
  for (const s of window) m += s.arousal;
  return m / window.length;
}

function valenceMean(window: ReadonlyArray<LongWindowSample>): number {
  if (window.length === 0) return 0;
  let m = 0;
  for (const s of window) m += s.valence;
  return m / window.length;
}

// Affect dynamics oscillation magnitude: mean absolute frame-to-frame change
// in V (Kuppens et al. 2010, 2013, Emotion 10/13). High = stress / lability;
// low = calm.
function valenceFlux(window: ReadonlyArray<LongWindowSample>): number {
  if (window.length < 2) return 0;
  let acc = 0;
  for (let i = 1; i < window.length; i++) {
    acc += Math.abs(window[i]!.valence - window[i - 1]!.valence);
  }
  return acc / (window.length - 1);
}

// Compute per-user resting metrics from a window of samples captured during
// the 3 s neutral baseline calibration. Caller should supply samplesPerSec
// computed from actual inference timestamps, since hardware varies.
export function computeUserBaseline(
  window: ReadonlyArray<LongWindowSample>,
  samplesPerSec: number,
): UserBaseline {
  return {
    blinkRate: blinkRatePerSec(window, samplesPerSec),
    intensityVariance: intensityVariance(window),
    perclos: perclos(window),
    valence: valenceMean(window),
    arousal: arousalMean(window),
    valenceFlux: valenceFlux(window),
  };
}

// ---- State scores (each independently in [0,1]) ----

export function computeStateScores(
  input: StateInputs,
): Record<CognitiveState, number> {
  const bs = input.blendshapes;
  const raw = input.rawBlendshapes;
  const eyeSquint = avg2(get(bs, "eyeSquintLeft"), get(bs, "eyeSquintRight"));
  const eyeWide = avg2(get(bs, "eyeWideLeft"), get(bs, "eyeWideRight"));
  const browDown = avg2(get(bs, "browDownLeft"), get(bs, "browDownRight"));
  const browInnerUp = get(bs, "browInnerUp");
  const browOuterUp = avg2(get(bs, "browOuterUpLeft"), get(bs, "browOuterUpRight"));
  const mouthPress = avg2(get(bs, "mouthPressLeft"), get(bs, "mouthPressRight"));
  const mouthStretch = avg2(get(bs, "mouthStretchLeft"), get(bs, "mouthStretchRight"));
  const jawForward = get(bs, "jawForward");
  const jawOpen = get(bs, "jawOpen");
  const smile = avg2(get(bs, "mouthSmileLeft"), get(bs, "mouthSmileRight"));

  const perclosVal = perclos(input.longWindow);
  const blinkRate = blinkRatePerSec(input.longWindow, input.samplesPerSec);
  const intVar = intensityVariance(input.longWindow);
  const arMean = arousalMean(input.longWindow);
  const vMean = valenceMean(input.longWindow);
  const vFlux = valenceFlux(input.longWindow);
  const rawEyeBlink = avg2(get(raw, "eyeBlinkLeft"), get(raw, "eyeBlinkRight"));

  // User baselines with literature-defaulting fallbacks. Each guard prevents
  // a degenerate baseline (e.g. 3 s neutral with one accidental blink → blink
  // rate looks like 1.0 Hz) from flipping every state ranking. We bound the
  // baseline to a plausible range and shrink toward the literature mean when
  // the sample is thin.
  const ub = input.userBaseline;
  // Bound the per-user blink rate to a plausible 0.05..1.2 Hz range
  // (Stern 1984 adult resting median is 0.25 Hz; clinical max for a healthy
  // adult under stress/fatigue is ~1 Hz). Guards against a 3 s baseline
  // capture that happened to contain an outlier blink count.
  const personalBlink = ub
    ? Math.min(1.2, Math.max(0.05, ub.blinkRate))
    : FALLBACK_BLINK_RATE_HZ;
  const personalIntVar = ub
    ? Math.max(0.001, ub.intensityVariance)
    : 0.01;
  // Resting PERCLOS is typically near 0 in awake individuals (Dinges &
  // Grace 1998); cap at 0.10 to prevent a noisy baseline from raising the
  // tired threshold past the meaningful alarm point.
  const personalPerclos = ub
    ? Math.max(0, Math.min(0.10, ub.perclos))
    : 0.02;
  const personalArousal = ub ? ub.arousal : 0;
  const personalValence = ub ? ub.valence : 0;
  const personalVFlux = ub ? Math.max(0.005, ub.valenceFlux) : 0.02;

  // Deviations from the user's resting baseline -- positive = "above usual".
  const dPerclos = perclosVal - personalPerclos;
  const dBlink = blinkRate - personalBlink;
  const dArousal = arMean - personalArousal;
  const dValence = vMean - personalValence;
  const dVFlux = vFlux - personalVFlux;

  // Coefficients below are tuned for stability + literature alignment:
  //  - The dominant signal for each state stays high-coefficient so the
  //    state can clearly fire on its canonical pattern.
  //  - Secondary signals stay low-coefficient so single-frame jitter (a
  //    blink, a yawn, a mouth movement during speech) doesn't whiplash
  //    the score. Temporal smoothing applied on top in main.ts further
  //    reduces frame-to-frame swing.
  //  - Variance-of-intensity terms use a moderate coefficient (was 50,
  //    now 25) because raw variance varies in [0..0.05] and was making
  //    the focused/bored/calm states swing on small movements.
  //  - V-flux coefficients reduced (was 8, now 4) for the same reason.
  //  - Blink-rate coefficients reduced (was 2.0, now 1.0) because
  //    blink-rate over short windows is noisy.

  // ---- TIRED / DROWSY ----
  // Dominant: PERCLOS exceeding the canonical 0.15 threshold + the user's
  // own resting PERCLOS. Wierwille 1994; Dinges & Grace 1998 FHWA TB98-006;
  // Abe 2023 Sleep Advances.
  const tiredThresh = 0.15 + personalPerclos;
  const tired = sigmoid(
    6 * (perclosVal - tiredThresh) +
    0.4 * Math.max(0, dBlink) +
    0.5 * Math.max(0, -dArousal) +
    0.3 * rawEyeBlink -
    0.4 * input.intensityNorm -
    0.5 * smile,
  );

  // ---- FOCUSED / CONCENTRATING ----
  // Sustained AU4 + suppressed blink rate + low expression variance.
  // Stern 1984; Maffei & Angrilli 2018.
  const focused = sigmoid(
    1.2 * browDown +
    0.4 * eyeSquint +
    1.0 * Math.max(0, -dBlink) +
    25 * Math.max(0, personalIntVar * 0.7 - intVar) -
    0.6 * smile -
    0.5 * jawOpen -
    0.6 * eyeWide,
  );

  // ---- BORED ----
  // Expressive flatness vs personal variance + low arousal vs personal
  // baseline + AU26 without smile. D'Mello & Graesser 2010.
  const bored = sigmoid(
    0.7 * (1 - input.intensityNorm) +
    0.6 * Math.max(0, -dArousal) +
    25 * Math.max(0, personalIntVar * 0.8 - intVar) +
    0.4 * Math.max(0, jawOpen - smile) -
    0.5 * eyeSquint -
    0.4 * browInnerUp -
    0.4 * eyeWide,
  );

  // ---- STRESSED / ANXIOUS ----
  // AU1+AU2+AU4 fear-blend, AU20, AU24, blink/V-flux elevated vs personal.
  // Harrigan & O'Connell 1996; Giannakakis et al. 2017.
  const stressed = sigmoid(
    0.8 * Math.min(browInnerUp + browOuterUp, browDown * 1.5) +
    0.9 * mouthStretch +
    0.7 * mouthPress +
    0.4 * jawForward +
    0.5 * Math.max(0, dBlink) +
    4 * Math.max(0, dVFlux) -
    0.6 * smile,
  );

  // ---- ENGAGED ----
  // Whitehill et al. 2014: positive V vs personal, sustained gaze, AU12/
  // AU1+2 spikes, moderate-to-high arousal vs personal.
  const engaged = sigmoid(
    0.9 * Math.max(0, dValence) +
    0.6 * Math.max(0, dArousal) +
    0.4 * input.intensityNorm +
    0.3 * (smile + browInnerUp) -
    0.5 * Math.max(0, dBlink - 0.3) -
    0.8 * Math.max(0, dPerclos),
  );

  // ---- CONFUSED ----
  // D'Mello & Graesser 2010, Cog Emotion 24(1):67-76: AU4+AU7 co-firing.
  // Discrete pattern, no personal-baseline adjustment.
  const browDownAsym = Math.abs(get(bs, "browDownLeft") - get(bs, "browDownRight"));
  const confused = sigmoid(
    1.8 * Math.min(browDown, eyeSquint) +
    0.5 * browInnerUp +
    0.6 * browDownAsym +
    (input.ambiguous ? 0.5 : 0) -
    0.5 * smile,
  );

  // ---- CALM ----
  // Russell 1980 + Kuppens 2013: V/A near personal baseline, low V flux
  // vs baseline, expression variance near baseline, blink rate near baseline.
  const calm = sigmoid(
    -1.0 * Math.abs(dArousal) +
    -0.4 * Math.max(0, -dValence) +
    5 * Math.max(0, personalIntVar - intVar) +
    -5 * Math.max(0, dVFlux) +
    -0.6 * Math.abs(dBlink) -
    0.6 * stressed,
  );

  return {
    engaged: clamp01(engaged),
    focused: clamp01(focused),
    tired: clamp01(tired),
    bored: clamp01(bored),
    stressed: clamp01(stressed),
    confused: clamp01(confused),
    calm: clamp01(calm),
  };
}

export function rankStates(
  scores: Record<CognitiveState, number>,
  n: number = 2,
): ReadonlyArray<{ state: CognitiveState; score: number }> {
  return STATES.map((s) => ({ state: s, score: scores[s] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n);
}

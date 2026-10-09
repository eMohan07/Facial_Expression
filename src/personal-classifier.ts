// Personal emotion classifier trained from the user's calibration snapshots.
//
// Approach: diagonal-Gaussian Naive Bayes with a *shared* per-feature variance
// pooled across classes. Closed-form fit, no SGD. Works well on tiny data
// (~30 samples per class × 52 features) where SGD-based logistic regression
// would either over-fit or need carefully tuned regularization.
//
// log p(k | x) ∝  -0.5 * Σ_i (x_i - μ_ki)² / σ_i²   +   log π_k
//
// Features are baseline-subtracted so the "neutral" class lives near origin
// and per-class means represent muscle deltas from the user's resting face.

import { type Emotion } from "./emotion-head";

export interface EmotionTemplate {
  // Mean blendshape vector for the calibration window (display use).
  blendshapes: Record<string, number>;
  // Per-frame raw snapshots from the 3s window. Used for classifier training;
  // intra-class variance lets the model learn which features actually matter
  // for this user vs which are noise.
  samples: ReadonlyArray<Record<string, number>>;
  capturedAt: number;
}

export type PersonalTemplates = Partial<Record<Emotion, EmotionTemplate>>;

export interface PersonalProb {
  emotion: Emotion;
  prob: number;
  // For display / explanations: log-likelihood normalized to [0,1] by the
  // best-class score so the top-1's "raw" score is always 1.0 -- the others
  // show how much further apart they are.
  rel: number;
}

export interface PersonalClassifier {
  predict(currentBlendshapes: Record<string, number>): PersonalProb[];
  classes: ReadonlyArray<Emotion>;
}

const VAR_EPSILON = 1e-4;

function vecApplyBaseline(
  raw: Record<string, number>,
  baseline: Record<string, number> | null,
  keys: string[],
): Float32Array {
  const v = new Float32Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i]!;
    v[i] = (raw[k] ?? 0) - (baseline ? baseline[k] ?? 0 : 0);
  }
  return v;
}

interface TrainingExample {
  blendshapes: Record<string, number>;
  label: Emotion;
}

// Trains a personal classifier from labeled blendshape snapshots. Returns
// null when there are fewer than 2 distinct classes -- a single-class model
// would always pick that class with 100% confidence and tell the user
// nothing.
export function trainPersonalClassifier(
  examples: ReadonlyArray<TrainingExample>,
  baseline: Record<string, number> | null,
): PersonalClassifier | null {
  if (examples.length === 0) return null;

  const classSet = new Set<Emotion>();
  for (const ex of examples) classSet.add(ex.label);
  if (classSet.size < 2) return null;

  // Stable feature index: union of keys across all samples + baseline.
  const keySet = new Set<string>();
  for (const ex of examples) for (const k of Object.keys(ex.blendshapes)) keySet.add(k);
  if (baseline) for (const k of Object.keys(baseline)) keySet.add(k);
  keySet.delete("_neutral");
  const keys = [...keySet];
  const F = keys.length;

  const classes = [...classSet];
  const C = classes.length;
  const classIdx = new Map<Emotion, number>();
  classes.forEach((c, i) => classIdx.set(c, i));

  // Vectorize all examples once.
  const vectors: Float32Array[] = [];
  const labels: number[] = [];
  for (const ex of examples) {
    vectors.push(vecApplyBaseline(ex.blendshapes, baseline, keys));
    labels.push(classIdx.get(ex.label)!);
  }

  // Per-class means.
  const counts = new Int32Array(C);
  const means = new Float32Array(C * F);
  for (let n = 0; n < vectors.length; n++) {
    const v = vectors[n]!;
    const c = labels[n]!;
    counts[c]!++;
    for (let i = 0; i < F; i++) means[c * F + i]! += v[i]!;
  }
  for (let c = 0; c < C; c++) {
    const cnt = counts[c] || 1;
    for (let i = 0; i < F; i++) means[c * F + i]! /= cnt;
  }

  // Pooled per-feature variance: sum of (x - μ_class)² / N over all samples.
  // Shared across classes -- with limited per-class samples, per-class
  // variances would be unstable.
  const variance = new Float32Array(F);
  for (let n = 0; n < vectors.length; n++) {
    const v = vectors[n]!;
    const c = labels[n]!;
    for (let i = 0; i < F; i++) {
      const d = v[i]! - means[c * F + i]!;
      variance[i]! += d * d;
    }
  }
  for (let i = 0; i < F; i++) {
    variance[i] = variance[i]! / vectors.length + VAR_EPSILON;
  }
  const invVar = new Float32Array(F);
  for (let i = 0; i < F; i++) invVar[i] = 1 / variance[i]!;

  return {
    classes,
    predict(currentBlendshapes) {
      const x = vecApplyBaseline(currentBlendshapes, baseline, keys);
      const logp = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        let acc = 0;
        for (let i = 0; i < F; i++) {
          const d = x[i]! - means[c * F + i]!;
          acc += d * d * invVar[i]!;
        }
        logp[c] = -0.5 * acc;
      }

      // Softmax with the standard log-sum-exp shift.
      let maxLogp = -Infinity;
      for (let c = 0; c < C; c++) if (logp[c]! > maxLogp) maxLogp = logp[c]!;
      let sum = 0;
      const probs = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        const e = Math.exp(logp[c]! - maxLogp);
        probs[c] = e;
        sum += e;
      }
      for (let c = 0; c < C; c++) probs[c]! /= sum;

      const result: PersonalProb[] = [];
      for (let c = 0; c < C; c++) {
        result.push({
          emotion: classes[c]!,
          prob: probs[c]!,
          rel: Math.exp(logp[c]! - maxLogp),
        });
      }
      result.sort((a, b) => b.prob - a.prob);
      return result;
    },
  };
}

export function topTemplateBlendshapes(
  template: EmotionTemplate,
  baseline: Record<string, number> | null,
  n: number = 3,
): ReadonlyArray<{ name: string; score: number }> {
  const adjusted: Array<{ name: string; score: number }> = [];
  for (const k of Object.keys(template.blendshapes)) {
    if (k === "_neutral") continue;
    const v = (template.blendshapes[k] ?? 0) - (baseline ? baseline[k] ?? 0 : 0);
    if (v > 0) adjusted.push({ name: k, score: v });
  }
  return adjusted.sort((a, b) => b.score - a.score).slice(0, n);
}

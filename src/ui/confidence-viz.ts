// Top-2 ambiguity treatment (Step 6b) + temporal stability sparkline (Step 6c)
// + top-3 active blendshapes panel (Step 6e) + DTM14 compound emotion labels.

import { type Emotion } from "../emotion-head";

export const TOP2_AMBIGUITY_THRESHOLD = 0.15;
export const SPARKLINE_FRAMES = 60; // ~2s of inference samples
export const COMPOUND_SUSTAIN_MS = 1000;

// Compound prototypes for every pairwise combination of the 8 basic classes.
// DTM14 (Du, Tao & Martinez 2014) covers the canonical compounds among Ekman-6
// + happy/sad/disgust crosses; the rest extend beyond DTM14 into colloquial
// English labels for contempt-based and mild (neutral-paired) combinations.
// Keys are alphabetically-canonicalized "a|b" pairs so order doesn't matter.
const COMPOUND_LABELS: Record<string, string> = {
  // anger ×
  "angry|contempt": "scornful",
  "angry|disgust": "angrily disgusted",
  "angry|fear": "fearfully angry",
  "angry|happy": "triumphant",
  "angry|neutral": "tense",
  "angry|sad": "sadly angry",
  "angry|surprised": "angrily surprised",
  // contempt ×
  "contempt|disgust": "disdainful",
  "contempt|fear": "suspicious",
  "contempt|happy": "smug",
  "contempt|neutral": "skeptical",
  "contempt|sad": "resentful",
  "contempt|surprised": "incredulous",
  // disgust ×
  "disgust|fear": "fearfully disgusted",
  "disgust|happy": "happily disgusted",
  "disgust|neutral": "uneasy",
  "disgust|sad": "sadly disgusted",
  "disgust|surprised": "disgustedly surprised",
  // fear ×
  "fear|happy": "nervous excitement",
  "fear|neutral": "anxious",
  "fear|sad": "sadly fearful",
  "fear|surprised": "fearfully surprised",
  // happy ×
  "happy|neutral": "content",
  "happy|sad": "bittersweet",
  "happy|surprised": "happily surprised",
  // remaining neutral ×
  "neutral|sad": "subdued",
  "neutral|surprised": "perplexed",
  // sad ×
  "sad|surprised": "sadly surprised",
};

export function compoundLabel(e1: Emotion, e2: Emotion): string | null {
  const [a, b] = e1 < e2 ? [e1, e2] : [e2, e1];
  return COMPOUND_LABELS[`${a}|${b}`] ?? null;
}

// Top-N active blendshapes for the active-face panel (Step 6e). Filters out
// the "_neutral" channel which always sits near 1 when the face is at rest,
// so it would otherwise dominate the list and be useless as an "active" cue.
export function topActiveBlendshapes(
  bs: Record<string, number>,
  n: number = 3,
): ReadonlyArray<{ name: string; score: number }> {
  return Object.entries(bs)
    .filter(([k]) => k !== "_neutral")
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([name, score]) => ({ name, score }));
}

export function isAmbiguous(probs: number[]): boolean {
  const sorted = [...probs].sort((a, b) => b - a);
  return sorted.length >= 2 && sorted[0]! - sorted[1]! < TOP2_AMBIGUITY_THRESHOLD;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

// Sparkline of top-1 confidence over the last SPARKLINE_FRAMES samples.
// Flat line = stable prediction; jagged line = the model is changing its mind.
// `values` is plotted left (oldest) to right (newest); fewer than the buffer's
// max length is fine -- the line scales to fit what's there.
export function drawSparkline(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  values: ArrayLike<number>,
  color: string,
): void {
  if (values.length < 2) return;
  ctx.save();

  // Subtle gridline at 0.5 for visual reference.
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y + h * 0.5);
  ctx.lineTo(x + w, y + h * 0.5);
  ctx.stroke();

  // The actual line.
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  ctx.beginPath();
  const denom = Math.max(1, values.length - 1);
  for (let i = 0; i < values.length; i++) {
    const px = x + (i / denom) * w;
    const py = y + h - clamp01(values[i]!) * h;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  ctx.restore();
}

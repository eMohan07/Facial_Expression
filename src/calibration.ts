// Temperature scaling (Guo et al. 2017). Single scalar tau, fit once
// offline on a held-out validation set via LBFGS, hardcoded here.
// Re-fit tau whenever emotion-head.ts changes architecture.

// Placeholder until Step 5. HSEmotion logits are already cross-entropy-trained
// on AffectNet so τ=1 is a reasonable starting point; the real fit is via LBFGS
// (Guo et al. 2017) on a held-out validation set. Commit the reliability diagram
// to the README when τ is tuned.
export const TAU = 1.0;

export function calibratedSoftmax(logits: number[], tau = TAU): number[] {
  const scaled = logits.map((l) => l / tau);
  const m = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

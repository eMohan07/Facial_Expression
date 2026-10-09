// Valence/arousal 2D plot with 2s fading trail (Step 6d).
// Russell circumplex layout: valence on X (-1 left = negative, +1 right = positive),
// arousal on Y (+1 top = high, -1 bottom = low/sleepy).
// More honest to affective psychology than discrete emotion bars.

export interface VAPoint {
  v: number;
  a: number;
  t: number; // timestamp ms (performance.now())
}

export const VA_TRAIL_MS = 2000;

const clamp = (x: number, lo: number, hi: number): number =>
  x < lo ? lo : x > hi ? hi : x;

export function drawValenceArousal(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  trail: ReadonlyArray<VAPoint>,
  nowMs: number,
): void {
  ctx.save();
  // Counter-flip the X axis so the chart reads left-to-right under the
  // CSS-mirrored canvas (matches the readout-panel convention).
  ctx.translate(x + size, y);
  ctx.scale(-1, 1);

  // Panel background.
  ctx.fillStyle = "rgba(15, 15, 20, 0.78)";
  ctx.fillRect(0, 0, size, size);

  const inset = 10;
  const cx = size / 2;
  const cy = size / 2;
  const r = cx - inset;

  // Circumplex circle.
  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // Crosshair axes.
  ctx.beginPath();
  ctx.moveTo(inset, cy);
  ctx.lineTo(size - inset, cy);
  ctx.moveTo(cx, inset);
  ctx.lineTo(cx, size - inset);
  ctx.stroke();

  // Axis labels — counter-flip again locally so text reads correctly.
  ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  drawFlippedText(ctx, "+A", cx + 3, inset + 8);
  drawFlippedText(ctx, "-A", cx + 3, size - inset - 1);
  drawFlippedText(ctx, "+V", size - inset - 14, cy - 3);
  drawFlippedText(ctx, "-V", inset + 2, cy - 3);

  // Trail (oldest faded → newest opaque).
  const cutoff = nowMs - VA_TRAIL_MS;
  for (let i = 0; i < trail.length; i++) {
    const p = trail[i]!;
    if (p.t < cutoff) continue;
    const fade = 1 - (nowMs - p.t) / VA_TRAIL_MS;
    const px = cx + clamp(p.v, -1, 1) * r;
    const py = cy - clamp(p.a, -1, 1) * r;
    ctx.fillStyle = `rgba(165, 180, 252, ${(fade * 0.55).toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(px, py, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // Current point on top.
  if (trail.length > 0) {
    const p = trail[trail.length - 1]!;
    const px = cx + clamp(p.v, -1, 1) * r;
    const py = cy - clamp(p.a, -1, 1) * r;
    ctx.fillStyle = "rgba(99, 102, 241, 0.9)";
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(224, 231, 255, 0.9)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.restore();
}

function drawFlippedText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(-1, 1);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

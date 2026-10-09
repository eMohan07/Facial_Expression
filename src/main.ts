// Render-loop entry point. This is the only place requestAnimationFrame is
// called; everything else is pure functions over feature data.

import { createFacePipeline, type FacePipeline, type FaceFeatures } from "./face-pipeline";
import { createEmotionClassifier, type EmotionClassifier } from "./emotion-onnx";
import { EMOTIONS, type Emotion, type EmotionReadout } from "./emotion-head";
import { calibratedSoftmax } from "./calibration";
import {
  TOP2_AMBIGUITY_THRESHOLD,
  SPARKLINE_FRAMES,
  COMPOUND_SUSTAIN_MS,
  drawSparkline,
  topActiveBlendshapes,
  compoundLabel,
} from "./ui/confidence-viz";
import {
  VA_TRAIL_MS,
  drawValenceArousal,
  type VAPoint,
} from "./ui/valence-arousal-viz";
import {
  streamChat,
  buildPrompt,
  type ExplainSnapshot,
  type VATrajectoryPoint,
  type EmotionSegment,
  type ChatMessage,
} from "./ai/explain";
import {
  trainPersonalClassifier,
  topTemplateBlendshapes,
  type PersonalClassifier,
  type PersonalProb,
  type PersonalTemplates,
} from "./personal-classifier";
import {
  computeStateScores,
  computeUserBaseline,
  rankStates,
  STATES,
  STATES_MIN_SAMPLES,
  type CognitiveState,
  type LongWindowSample,
  type UserBaseline,
} from "./states";
import { GazeTracker, type GazeResult } from "./gaze-tracker";
import { DocumentChatController } from "./ui/document-chat";
import { FeatureDashboard } from "./ui/feature-dashboard";

const TARGET_WIDTH = 1280;
const TARGET_HEIGHT = 720;

interface Refs {
  videoContainer: HTMLElement;
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  pauseBtn: HTMLButtonElement;
  explainBtn: HTMLButtonElement;
  recalibrateBtn: HTMLButtonElement;
  exportBtn: HTMLButtonElement;
  recordBtn: HTMLButtonElement;
  recordDot: HTMLElement;
  recordLabel: HTMLElement;
  explanation: HTMLElement;
  chatForm: HTMLFormElement;
  chatInput: HTMLInputElement;
  chatSend: HTMLButtonElement;
  summarizeSession: HTMLButtonElement;
  discussRecording: HTMLButtonElement;
  loading: HTMLElement;
  loadingMsg: HTMLElement;
  personalCalibToggle: HTMLButtonElement;
  personalCalibPanel: HTMLElement;
  personalCalibGrid: HTMLElement;
  personalCalibOverlay: HTMLElement;
  personalCalibEmotion: HTMLElement;
  personalCalibCountdown: HTMLElement;
  personalCalibProgress: HTMLElement;
  storageEnabled: HTMLInputElement;
  storageStatus: HTMLElement;
  storageClear: HTMLButtonElement;
  windowSlider: HTMLInputElement;
  windowValue: HTMLElement;
  helpToggle: HTMLButtonElement;
  helpModal: HTMLElement;
  helpClose: HTMLButtonElement;
  helpCloseBottom: HTMLButtonElement;
}

function getRefs(): Refs {
  const videoContainer = document.getElementById("video-container");
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("overlay");
  const pauseBtn = document.getElementById("pause");
  const explainBtn = document.getElementById("explain");
  const recalibrateBtn = document.getElementById("recalibrate");
  const exportBtn = document.getElementById("export");
  const recordBtn = document.getElementById("record");
  const recordDot = document.getElementById("record-dot");
  const recordLabel = document.getElementById("record-label");
  const explanation = document.getElementById("explanation");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatSend = document.getElementById("chat-send");
  const summarizeSession = document.getElementById("summarize-session");
  const discussRecording = document.getElementById("discuss-recording");
  const loading = document.getElementById("loading");
  const loadingMsg = document.getElementById("loading-msg");
  const personalCalibToggle = document.getElementById("personal-calib-toggle");
  const personalCalibPanel = document.getElementById("personal-calib-panel");
  const personalCalibGrid = document.getElementById("personal-calib-grid");
  const personalCalibOverlay = document.getElementById("personal-calib-overlay");
  const personalCalibEmotion = document.getElementById("personal-calib-emotion");
  const personalCalibCountdown = document.getElementById("personal-calib-countdown");
  const personalCalibProgress = document.getElementById("personal-calib-progress");
  const storageEnabled = document.getElementById("storage-enabled");
  const storageStatus = document.getElementById("storage-status");
  const storageClear = document.getElementById("storage-clear");
  const windowSlider = document.getElementById("window-slider");
  const windowValue = document.getElementById("window-value");
  const helpToggle = document.getElementById("help-toggle");
  const helpModal = document.getElementById("help-modal");
  const helpClose = document.getElementById("help-close");
  const helpCloseBottom = document.getElementById("help-close-bottom");
  if (
    !videoContainer ||
    !(video instanceof HTMLVideoElement) ||
    !(canvas instanceof HTMLCanvasElement) ||
    !(pauseBtn instanceof HTMLButtonElement) ||
    !(explainBtn instanceof HTMLButtonElement) ||
    !(recalibrateBtn instanceof HTMLButtonElement) ||
    !(exportBtn instanceof HTMLButtonElement) ||
    !(recordBtn instanceof HTMLButtonElement) ||
    !recordDot ||
    !recordLabel ||
    !explanation ||
    !(chatForm instanceof HTMLFormElement) ||
    !(chatInput instanceof HTMLInputElement) ||
    !(chatSend instanceof HTMLButtonElement) ||
    !(summarizeSession instanceof HTMLButtonElement) ||
    !(discussRecording instanceof HTMLButtonElement) ||
    !loading ||
    !loadingMsg ||
    !(personalCalibToggle instanceof HTMLButtonElement) ||
    !personalCalibPanel ||
    !personalCalibGrid ||
    !personalCalibOverlay ||
    !personalCalibEmotion ||
    !personalCalibCountdown ||
    !personalCalibProgress ||
    !(storageEnabled instanceof HTMLInputElement) ||
    !storageStatus ||
    !(storageClear instanceof HTMLButtonElement) ||
    !(windowSlider instanceof HTMLInputElement) ||
    !windowValue ||
    !(helpToggle instanceof HTMLButtonElement) ||
    !helpModal ||
    !(helpClose instanceof HTMLButtonElement) ||
    !(helpCloseBottom instanceof HTMLButtonElement)
  ) {
    throw new Error("missing required DOM nodes");
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  return {
    videoContainer,
    video, canvas, ctx, pauseBtn, explainBtn, recalibrateBtn, exportBtn, explanation,
    recordBtn, recordDot, recordLabel,
    chatForm, chatInput, chatSend, summarizeSession, discussRecording,
    loading, loadingMsg, personalCalibToggle, personalCalibPanel, personalCalibGrid,
    personalCalibOverlay, personalCalibEmotion, personalCalibCountdown, personalCalibProgress,
    storageEnabled, storageStatus, storageClear,
    windowSlider, windowValue,
    helpToggle, helpModal, helpClose, helpCloseBottom,
  };
}

function showLoading(refs: Refs, message: string): void {
  refs.loadingMsg.textContent = message;
  refs.loading.classList.remove("hidden");
  refs.loading.classList.add("flex");
}

function hideLoading(refs: Refs): void {
  refs.loading.classList.add("hidden");
  refs.loading.classList.remove("flex");
}

function showPersonalCalib(refs: Refs, emotion: Emotion): void {
  refs.personalCalibEmotion.textContent = emotion;
  refs.personalCalibCountdown.textContent = "3";
  refs.personalCalibProgress.style.width = "0%";
  refs.personalCalibOverlay.classList.remove("hidden");
  refs.personalCalibOverlay.classList.add("flex");
}

function updatePersonalCalib(
  refs: Refs,
  remainingSec: number,
  fractionDone: number,
): void {
  refs.personalCalibCountdown.textContent = String(Math.max(0, remainingSec));
  refs.personalCalibProgress.style.width = `${Math.min(100, fractionDone * 100).toFixed(1)}%`;
}

function hidePersonalCalib(refs: Refs): void {
  refs.personalCalibOverlay.classList.add("hidden");
  refs.personalCalibOverlay.classList.remove("flex");
}

async function startWebcam(
  video: HTMLVideoElement,
  container: HTMLElement,
): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: TARGET_WIDTH },
      height: { ideal: TARGET_HEIGHT },
      facingMode: "user",
    },
  });
  video.srcObject = stream;
  await video.play();
  await new Promise<void>((resolve) => {
    if (video.readyState >= 2) {
      resolve();
      return;
    }
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
  // Match container's aspect ratio to the actual stream so phone portrait
  // cameras (e.g. 720x1280) fill the box instead of being cropped or
  // letterboxed inside a forced 16:9 container. On laptops the webcam is
  // already 16:9 so this is a no-op for the desktop layout.
  const applyAspect = (): void => {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      container.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    }
  };
  applyAspect();
  video.addEventListener("loadedmetadata", applyAspect);
  window.addEventListener("orientationchange", applyAspect);
  return stream;
}

function sizeCanvasToVideo(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

interface DisplayReadout {
  top: { emotion: Emotion; prob: number }[];
  valence: number;
  arousal: number;
  ambiguous: boolean;
}

function toDisplay(r: EmotionReadout): DisplayReadout {
  const ordered = EMOTIONS.map((e) => r.logits[e]);
  const probs = calibratedSoftmax(ordered);
  const ranked = EMOTIONS.map((e, i) => ({ emotion: e, prob: probs[i]! }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
  const ambiguous =
    ranked.length >= 2 && ranked[0]!.prob - ranked[1]!.prob < TOP2_AMBIGUITY_THRESHOLD;
  return { top: ranked, valence: r.valence, arousal: r.arousal, ambiguous };
}

// Map top-1 confidence to bbox stroke alpha: bright at 90%+, translucent at 40%.
function confidenceAlpha(prob: number): number {
  const t = Math.max(0, Math.min(1, (prob - 0.4) / 0.5));
  return 0.3 + 0.65 * t;
}

interface StateWarmupInfo {
  samples: number;
  target: number;
  seconds: number;
}

function drawDebugBoxes(
  ctx: CanvasRenderingContext2D,
  faces: FaceFeatures[],
  primaryReadout: DisplayReadout | null,
  probHistory: number[],
  primaryBlendshapes: Record<string, number>,
  personalRanked: PersonalProb[] | null,
  compound: string | null,
  pulseReasonActive: string | null,
  nowMs: number,
  intensityNorm: number,
  stateScores: Record<CognitiveState, number> | null,
  stateWarmup: StateWarmupInfo | null,
): void {
  ctx.lineWidth = 3;
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i]!;
    const { bbox } = face;
    if (i === 0 && primaryReadout) {
      let a = confidenceAlpha(primaryReadout.top[0]!.prob);
      // Heartbeat-pulse the bbox alpha when something interesting is on.
      // 1.5 Hz feels deliberate without being distracting.
      if (pulseReasonActive) {
        const phase = (nowMs / 1000) * 2 * Math.PI * 1.5;
        a = 0.55 + 0.4 * (Math.sin(phase) * 0.5 + 0.5);
      }
      ctx.strokeStyle = `rgba(99, 102, 241, ${a.toFixed(3)})`;
    } else {
      ctx.strokeStyle = "rgba(99, 102, 241, 0.55)";
    }
    ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);

    // Visual iris tracking points
    if (face.landmarks && face.landmarks.length >= 478 * 3) {
      const lx = face.landmarks[468 * 3]! * ctx.canvas.width;
      const ly = face.landmarks[468 * 3 + 1]! * ctx.canvas.height;
      const rx = face.landmarks[473 * 3]! * ctx.canvas.width;
      const ry = face.landmarks[473 * 3 + 1]! * ctx.canvas.height;

      ctx.save();
      ctx.fillStyle = "rgba(56, 189, 248, 0.9)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(rx, ry, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    if (i === 0 && primaryReadout) {
      // Display & explain use baseline-subtracted blendshapes so we surface
      // *changes* from the user's resting face, not their neutral asymmetry.
      const topBs = topActiveBlendshapes(primaryBlendshapes, 3);
      drawReadoutPanel(
        ctx,
        bbox,
        primaryReadout,
        probHistory,
        topBs,
        personalRanked,
        compound,
        intensityNorm,
        stateScores,
        stateWarmup,
      );
      if (pulseReasonActive) {
        drawPulseBadge(ctx, bbox, pulseReasonActive, nowMs);
      }
    } else {
      drawSimpleLabel(ctx, bbox, `face ${i}`);
    }
  }
}

function drawPulseBadge(
  ctx: CanvasRenderingContext2D,
  bbox: { x: number; y: number; w: number; h: number },
  reason: string,
  nowMs: number,
): void {
  const phase = (nowMs / 1000) * 2 * Math.PI * 1.5;
  const ringAlpha = 0.4 + 0.4 * (Math.sin(phase) * 0.5 + 0.5);
  // Counter-flip so the badge anchors at the visible top-right of the face.
  ctx.save();
  ctx.translate(bbox.x, bbox.y);
  ctx.scale(-1, 1);
  // Anchor at x=0 in the flipped frame, then move into the badge corner.
  const cx = -16;
  const cy = -2;
  // Outer ring: pulses opacity.
  ctx.beginPath();
  ctx.arc(cx, cy, 13, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(251, 191, 36, ${ringAlpha.toFixed(3)})`;
  ctx.lineWidth = 2;
  ctx.stroke();
  // Solid amber dot.
  ctx.beginPath();
  ctx.arc(cx, cy, 9, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
  ctx.fill();
  ctx.fillStyle = "rgba(15, 15, 20, 0.95)";
  ctx.font = "bold 13px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.scale(-1, 1);
  ctx.fillText("?", -cx, cy);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
  ctx.scale(-1, 1);
  // Reason label below the badge.
  ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  const labelW = ctx.measureText(reason).width;
  ctx.scale(-1, 1);
  ctx.fillText(reason, -cx + labelW / 2, cy + 22);
  ctx.restore();
}

function drawSimpleLabel(
  ctx: CanvasRenderingContext2D,
  bbox: { x: number; y: number; w: number; h: number },
  text: string,
): void {
  ctx.save();
  ctx.translate(bbox.x + bbox.w, bbox.y - 6);
  ctx.scale(-1, 1);
  ctx.font = "14px system-ui, sans-serif";
  ctx.fillStyle = "rgba(99, 102, 241, 0.95)";
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function drawReadoutPanel(
  ctx: CanvasRenderingContext2D,
  bbox: { x: number; y: number; w: number; h: number },
  r: DisplayReadout,
  probHistory: number[],
  topBlendshapes: ReadonlyArray<{ name: string; score: number }>,
  personalRanked: PersonalProb[] | null,
  compound: string | null,
  intensityNorm: number,
  stateScores: Record<CognitiveState, number> | null,
  stateWarmup: StateWarmupInfo | null,
): void {
  const lineH = 16;
  const bsLineH = 14;
  const stateLineH = 14;
  const padX = 8;
  const padY = 6;
  const panelW = 188;
  const sparkH = 26;
  const intensityH = 18; // INTENSITY label + bar
  const headerH = r.ambiguous ? lineH : 0;
  const lines = r.top.length + 1;
  const topStatesList = stateScores ? rankStates(stateScores, 2) : [];
  const stateBlockH =
    topStatesList.length > 0
      ? lineH + topStatesList.length * stateLineH + 4
      : stateWarmup
        ? lineH + 4
        : 0;
  const bsBlockH = topBlendshapes.length > 0 ? lineH + topBlendshapes.length * bsLineH + 4 : 0;
  const personalH = personalRanked && personalRanked.length > 0 ? lineH + 4 : 0;
  const panelH = headerH + lines * lineH + intensityH + sparkH + stateBlockH + bsBlockH + personalH + padY * 2;

  ctx.save();
  // Anchor at top-right of bbox -- after the canvas mirror this becomes the
  // top-left of the visible face, which is where most viewers expect a label.
  ctx.translate(bbox.x + bbox.w, bbox.y);
  ctx.scale(-1, 1);

  ctx.fillStyle = "rgba(15, 15, 20, 0.78)";
  ctx.fillRect(0, 0, panelW, panelH);

  ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, monospace";

  let yCursor = padY;
  if (r.ambiguous) {
    // Highlight strip across the top so the user knows we're treating top-1
    // and top-2 as equally plausible. If the pair has been sustained long
    // enough and matches a DTM14 compound, name it instead of just saying
    // "ambiguous" -- a compound is the more honest reading.
    ctx.fillStyle = "rgba(251, 191, 36, 0.18)";
    ctx.fillRect(0, 0, panelW, headerH + padY);
    ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
    const headerText = compound ?? "ambiguous (top-2 close)";
    ctx.fillText(headerText, padX, padY + lineH - 4);
    yCursor += lineH;
  }

  for (let j = 0; j < r.top.length; j++) {
    const row = r.top[j]!;
    const y = yCursor + (j + 1) * lineH - 4;
    // Equalize emphasis on top-1 and top-2 when ambiguous; otherwise demote runners-up.
    const isHighlit = r.ambiguous ? j <= 1 : j === 0;
    ctx.fillStyle = isHighlit ? "rgba(165, 180, 252, 1)" : "rgba(203, 213, 225, 0.85)";
    ctx.fillText(row.emotion.padEnd(10, " "), padX, y);
    const pct = `${(row.prob * 100).toFixed(0)}%`;
    ctx.fillText(pct, panelW - padX - ctx.measureText(pct).width, y);
    const barX = padX + 76;
    const barW = panelW - barX - padX - 32;
    ctx.fillStyle = isHighlit ? "rgba(99, 102, 241, 0.65)" : "rgba(99, 102, 241, 0.25)";
    ctx.fillRect(barX, y - 10, barW * row.prob, 11);
  }

  ctx.fillStyle = "rgba(148, 163, 184, 0.9)";
  const vaText = `v ${r.valence.toFixed(2)}   a ${r.arousal.toFixed(2)}`;
  const vaY = yCursor + (r.top.length + 1) * lineH - 4;
  ctx.fillText(vaText, padX, vaY);

  // Expression intensity (independent of classification): how much muscle
  // activity is happening relative to the user's neutral baseline.
  const intY = vaY + 8;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
  ctx.fillText("INTENSITY", padX, intY + 7);
  const intBarX = padX + 64;
  const intBarW = panelW - intBarX - padX - 28;
  ctx.fillStyle = "rgba(99, 102, 241, 0.18)";
  ctx.fillRect(intBarX, intY, intBarW, 5);
  ctx.fillStyle = "rgba(165, 180, 252, 0.95)";
  ctx.fillRect(intBarX, intY, intBarW * intensityNorm, 5);
  const pct = `${Math.round(intensityNorm * 100)}%`;
  ctx.fillStyle = "rgba(203, 213, 225, 0.85)";
  ctx.fillText(pct, panelW - padX - ctx.measureText(pct).width, intY + 7);

  // Stability sparkline (Step 6c). Flat = stable, jittery = uncertain.
  const sparkY = intY + intensityH;
  drawSparkline(
    ctx,
    padX,
    sparkY,
    panelW - padX * 2,
    sparkH - 4,
    probHistory,
    "rgba(165, 180, 252, 0.95)",
  );

  // Top cognitive states (orthogonal to the 8 emotion classes). Hidden
  // until the long window has 60s of data per literature-defined feature
  // windows (PERCLOS: Wierwille 1994; blink rate: Stern 1984). Until then,
  // a small "warming up" countdown sits where states will appear.
  let cursor = sparkY + sparkH;
  if (topStatesList.length === 0 && stateWarmup) {
    const stY = cursor;
    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const remaining = Math.max(0, 60 - stateWarmup.seconds);
    ctx.fillText(`STATES warming up · ${remaining}s left`, padX, stY + 9);
    cursor = stY + lineH + 4;
  }
  if (topStatesList.length > 0) {
    let stY = cursor;
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText("STATES", padX, stY + 9);
    stY += 12;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let j = 0; j < topStatesList.length; j++) {
      const row = topStatesList[j]!;
      const ty = stY + (j + 1) * stateLineH - 3;
      ctx.fillStyle = j === 0
        ? "rgba(134, 239, 172, 0.95)"   // top state: emerald
        : "rgba(187, 247, 208, 0.85)";
      ctx.fillText(row.state, padX, ty);
      const v = `${(row.score * 100).toFixed(0)}%`;
      ctx.fillStyle = "rgba(203, 213, 225, 0.9)";
      ctx.fillText(v, panelW - padX - ctx.measureText(v).width, ty);
      const barX = padX + 70;
      const barW = panelW - barX - padX - 32;
      ctx.fillStyle = j === 0 ? "rgba(34, 197, 94, 0.55)" : "rgba(34, 197, 94, 0.25)";
      ctx.fillRect(barX, ty - 9, barW * row.score, 9);
    }
    cursor = stY + topStatesList.length * stateLineH + 4;
  }

  // Top active blendshapes (Step 6e). Grounds what the "Why?" prompt
  // feeds Claude.
  if (topBlendshapes.length > 0) {
    let bsY = cursor;
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText("ACTIVE", padX, bsY + 9);
    bsY += 12;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let j = 0; j < topBlendshapes.length; j++) {
      const row = topBlendshapes[j]!;
      const ty = bsY + (j + 1) * bsLineH - 3;
      ctx.fillStyle = "rgba(203, 213, 225, 0.9)";
      ctx.fillText(row.name, padX, ty);
      const v = row.score.toFixed(2);
      ctx.fillStyle = "rgba(165, 180, 252, 0.95)";
      ctx.fillText(v, panelW - padX - ctx.measureText(v).width, ty);
    }
    cursor = bsY + topBlendshapes.length * bsLineH;
  }

  // Personal classifier pick (cosine similarity over user's templates).
  // Only shown when ≥2 templates are calibrated -- a single template would
  // always score itself at 100% and be misleading.
  if (personalRanked && personalRanked.length > 0) {
    const top = personalRanked[0]!;
    const ty = cursor + lineH - 4;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.fillText("PERSONAL", padX, ty);
    ctx.fillStyle = top.emotion === r.top[0]!.emotion
      ? "rgba(165, 243, 252, 0.95)"   // agreement: cyan
      : "rgba(251, 191, 36, 0.95)";   // disagreement with HSEmotion: amber
    const label = `${top.emotion} ${(top.prob * 100).toFixed(0)}%`;
    ctx.fillText(label, panelW - padX - ctx.measureText(label).width, ty);
  }

  ctx.restore();
}

function showError(refs: Refs, message: string): void {
  refs.explanation.textContent = message;
  refs.explanation.classList.add("text-red-300");
}

async function main(): Promise<void> {
  const refs = getRefs();

  // AI features (Why? / Ask / Summarize / Discuss recording) require either a
  // server-side proxy URL (production) or an embedded dev key (local only).
  // Without either, hide every AI-related affordance via body.no-ai → CSS.
  // The rest of the app — calibration, states, recording, export, personal
  // templates — works unchanged.
  const aiEnabled =
    !!import.meta.env.VITE_EXPLAIN_PROXY_URL ||
    !!import.meta.env.VITE_DEMO_ANTHROPIC_API_KEY;
  if (!aiEnabled) {
    document.body.classList.add("no-ai");
  }

  refs.explanation.textContent = aiEnabled
    ? "Click “Start” to enable your webcam. Frames stay on this device."
    : "";
  refs.pauseBtn.textContent = "Start";

  const gazeTracker = new GazeTracker();
  const docChat = new DocumentChatController();
  const dashboard = new FeatureDashboard();
  let lastGazeResult: GazeResult | null = null;

  let paused = false;
  let stream: MediaStream | null = null;
  let pipeline: FacePipeline | null = null;
  let classifier: EmotionClassifier | null = null;

  // Async inference state: render every frame, classify whenever the previous
  // call has resolved. Keeps the UI smooth even if inference is 30-60ms.
  let inferenceInFlight = false;
  let lastReadout: DisplayReadout | null = null;
  let lastTimestamp = -1;

  // Top-1 probability history for the primary face (Step 6c). Reset when no
  // faces are detected -- assume the next person to enter frame is fresh
  // (proper identity tracking is Step 8).
  const probHistory: number[] = [];
  // Valence/arousal trail (Step 6d). 2s rolling window in wall-clock ms.
  const vaTrail: VAPoint[] = [];

  // Full-session timeline for JSON export. Grows during the session, only
  // reset on Recalibrate. SESSION_MAX_SAMPLES is a safety cap, never expected
  // to hit during normal use.
  interface SessionSample {
    tSec: number;
    emotion: Emotion;
    prob: number;
    valence: number;
    arousal: number;
    intensity: number;
    topBlendshapes: ReadonlyArray<{ name: string; score: number }>;
  }
  const sessionTimeline: SessionSample[] = [];
  let sessionStartedAt = 0;
  const SESSION_MAX_SAMPLES = 36000; // ~hour at 10Hz

  // Auto-trigger pulse: nudge the user to click Why? when the readout has
  // become informative. Three reasons are checked on each new inference:
  // sustained ambiguity, big V/A jump, and HSEmotion-vs-personal disagreement.
  // Cooldown prevents constant pulsing during a long ambiguous stretch.
  const AMBIGUITY_INTERESTING_MS = 3000;
  const VA_JUMP_THRESHOLD = 0.4;
  const PULSE_DURATION_MS = 5000;
  const PULSE_COOLDOWN_MS = 3000;
  let prevVA: { v: number; a: number } | null = null;
  let pulseUntilMs = 0;
  let pulseCooldownUntilMs = 0;
  let pulseReason: string | null = null;

  // Rolling 30s timeline (max). The Why? window slider clips to [2, 30]s when
  // building the snapshot. We keep the full 30s in the buffer so the slider is
  // immediate -- no waiting to "fill" a longer window.
  const TIMELINE_MS = 30000;
  let whyWindowSec = 5;
  interface TimelineSample {
    t: number;
    emotion: Emotion;
    prob: number;
    valence: number;
    arousal: number;
  }
  const timeline: TimelineSample[] = [];

  // Latest blendshape snapshot for the primary face (baseline-subtracted),
  // used by the readout panel and the "Why?" prompt.
  let latestBlendshapes: Record<string, number> = {};

  // Cognitive-state scores. Recomputed each readout from current blendshapes
  // + a 60s long window for PERCLOS, blink rate, expression entropy, V/A
  // dynamics. See src/states.ts for the citation-backed feature definitions.
  // Smoothed via EMA before display: per-frame state logits are noisy on
  // short windows (blink rate, intensity variance both jitter), and the
  // user reads "states" as something steadier than "current emotion".
  // Time constant: alpha=0.08 → ~12-sample window ≈ 1.2 s at 10 Hz.
  const STATE_SMOOTHING_ALPHA = 0.08;
  let lastStateScores: Record<CognitiveState, number> | null = null;
  let smoothedStateScores: Record<CognitiveState, number> | null = null;

  // Live samples-per-second estimate for the warmup countdown. We use the
  // same estimator the state features use; reading from longWindowTimes is
  // safe because by the time this runs, the long window has at least 1
  // sample (otherwise we wouldn't be in this branch).
  const samplesPerSecLive = (): number =>
    longWindowTimes.length >= 2
      ? (longWindowTimes.length - 1) /
        Math.max(0.5, (longWindowTimes[longWindowTimes.length - 1]! - longWindowTimes[0]!) / 1000)
      : 10;
  const longWindow: LongWindowSample[] = [];
  const LONG_WINDOW_SAMPLES = 600; // ~60s at 10Hz inference
  // Track inference timestamps in a small ring so we can estimate samplesPerSec
  // empirically rather than hardcoding 10Hz. PERCLOS / blink rate are sensitive
  // to this — actual rate varies with hardware.
  const longWindowTimes: number[] = [];
  // Samples from the 3 s neutral calibration window. Used to compute the
  // user's personal resting metrics (blink rate, expression flatness, etc.)
  // that calibrate the state thresholds to *their* normal.
  const baselineSamples: LongWindowSample[] = [];
  const baselineSampleTimes: number[] = [];
  let userBaseline: UserBaseline | null = null;

  // Aborts an in-flight explanation when the user re-clicks "Why?".
  let explainAbort: AbortController | null = null;

  // Personal baseline calibration. We capture ~3s of neutral resting face at
  // session start, average per-blendshape, and subtract that floor from every
  // subsequent frame's blendshapes. Cancels MediaPipe's resting-face L/R
  // asymmetry (~40% per github issues) and individual face geometry, so
  // "ACTIVE: mouthSmileLeft 0.85" no longer fires on a neutral, slightly
  // crooked face.
  const CALIB_DURATION_MS = 3000;
  let baseline: Record<string, number> | null = null;
  let calibStartedAt = 0;
  // Per-frame raw blendshape snapshots collected during the calibration window.
  // Used both to compute the mean (= the baseline / template means for display)
  // and as labeled training data for the personal classifier.
  let calibFrames: Record<string, number>[] = [];
  // Frames from the resting-face baseline capture, kept around so the personal
  // classifier always has a "neutral" class.
  let neutralFrames: Record<string, number>[] = [];

  const meanFrame = (frames: Record<string, number>[]): Record<string, number> => {
    if (frames.length === 0) return {};
    const sum: Record<string, number> = {};
    for (const f of frames) {
      for (const k in f) sum[k] = (sum[k] ?? 0) + (f[k] ?? 0);
    }
    const out: Record<string, number> = {};
    for (const k in sum) out[k] = sum[k]! / frames.length;
    return out;
  };
  const applyBaseline = (raw: Record<string, number>): Record<string, number> => {
    if (!baseline) return raw;
    const out: Record<string, number> = {};
    for (const k in raw) {
      const v = (raw[k] ?? 0) - (baseline[k] ?? 0);
      out[k] = v > 0 ? v : 0;
    }
    return out;
  };
  const startRecalibration = (): void => {
    baseline = null;
    calibStartedAt = 0;
    calibFrames = [];
    neutralFrames = [];
    personalClassifier = null;
    sessionTimeline.length = 0;
    sessionStartedAt = 0;
    prevVA = null;
    pulseUntilMs = 0;
    pulseCooldownUntilMs = 0;
    pulseReason = null;
    baselineSamples.length = 0;
    baselineSampleTimes.length = 0;
    userBaseline = null;
    smoothedStateScores = null;
  };

  // L2 norm of the baseline-subtracted blendshape vector. Independent of
  // emotion classification: a higher number means more facial activity right
  // now, regardless of which emotion (or whether the model is sure).
  const intensityFromAdjusted = (adjusted: Record<string, number>): number => {
    let sumSq = 0;
    for (const k in adjusted) {
      const v = adjusted[k] ?? 0;
      sumSq += v * v;
    }
    return Math.sqrt(sumSq);
  };
  // Empirical: a clear apex expression on a single AU sits ~1.0; a wide,
  // multi-AU expression can hit ~2.5. Normalize by 2.5 → display [0,1].
  const INTENSITY_NORM = 2.5;

  // -- Session JSON export. Reuses the in-memory session timeline; no new
  // privacy surface. Writes a versioned schema so future readers can detect
  // shape changes.
  const exportSession = (): void => {
    const payload = {
      schemaVersion: 1,
      exportedAt: new Date().toISOString(),
      sessionStartedAtMs:
        sessionStartedAt > 0 ? Date.now() - (performance.now() - sessionStartedAt) : null,
      sessionDurationSec:
        sessionStartedAt > 0 ? (performance.now() - sessionStartedAt) / 1000 : 0,
      sampleCount: sessionTimeline.length,
      samples: sessionTimeline,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.download = `expression-session-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // -- Personal emotion templates (opt-in). Captures 3s of the user
  // making each emotion's face and stores the per-blendshape average. Stored
  // in memory only; lost on refresh. Used for: a personal cosine-similarity
  // classifier shown alongside HSEmotion, comparison row in the panel, and
  // extra grounding in the "Why?" prompt.
  const personalTemplates: PersonalTemplates = {};
  let personalCalibState: {
    emotion: Emotion;
    startedAt: number;
    frames: Record<string, number>[];
  } | null = null;
  // The trained personal classifier (diagonal Gaussian) — null until ≥2 emotion
  // templates have been captured.
  let personalClassifier: PersonalClassifier | null = null;
  // Latest personal classifier output (recomputed on each new HSEmotion readout).
  let lastPersonalRanked: PersonalProb[] | null = null;

  // DTM14 compound emotion tracking. We require the same top-2 pair for
  // COMPOUND_SUSTAIN_MS before promoting the ambiguity strip to a compound
  // label, so a single noisy frame doesn't trigger it.
  let ambiguityRun: { pair: string; emotion1: Emotion; emotion2: Emotion; sinceMs: number } | null = null;

  // Rebuild the classifier from neutral frames + every calibrated emotion's
  // frame samples. Tiny model + closed-form fit means this runs in <5ms even
  // with a few hundred samples; safe to call on every new template.
  const retrainPersonalClassifier = (): void => {
    if (!baseline) {
      personalClassifier = null;
      return;
    }
    const examples: Array<{ blendshapes: Record<string, number>; label: Emotion }> = [];
    for (const f of neutralFrames) examples.push({ blendshapes: f, label: "neutral" });
    for (const e of EMOTIONS) {
      const tmpl = personalTemplates[e];
      if (!tmpl) continue;
      for (const f of tmpl.samples) examples.push({ blendshapes: f, label: e });
    }
    personalClassifier = trainPersonalClassifier(examples, baseline);
  };

  const onPersonalCalibrate = (e: Emotion): void => {
    if (personalCalibState) return; // a capture is already running
    if (!baseline) {
      // Wait for the neutral baseline first to avoid the user calibrating
      // their personal "happy" while the system is still in resting-face mode.
      return;
    }
    if (personalTemplates[e]) {
      const ok = window.confirm(
        `Replace your existing "${e}" template? This cannot be undone.`,
      );
      if (!ok) return;
    }
    personalCalibState = { emotion: e, startedAt: 0, frames: [] };
  };

  const renderPersonalCalibGrid = (): void => {
    refs.personalCalibGrid.innerHTML = "";
    // Skip neutral -- it's covered by the resting-face baseline above.
    for (const e of EMOTIONS) {
      if (e === "neutral") continue;
      const calibrated = !!personalTemplates[e];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = calibrated ? `${e} ✓` : e;
      btn.className = [
        "px-3 py-2 rounded text-sm border transition-colors",
        calibrated
          ? "border-indigo-700 bg-indigo-950/40 text-indigo-200 hover:bg-indigo-950/60"
          : "border-neutral-800 bg-neutral-950 text-neutral-300 hover:bg-neutral-800",
      ].join(" ");
      btn.addEventListener("click", () => onPersonalCalibrate(e));
      refs.personalCalibGrid.appendChild(btn);
    }
  };
  renderPersonalCalibGrid();

  refs.personalCalibToggle.addEventListener("click", () => {
    const willOpen = refs.personalCalibPanel.classList.contains("hidden");
    refs.personalCalibPanel.classList.toggle("hidden");
    refs.personalCalibToggle.setAttribute("aria-expanded", String(willOpen));
    refs.personalCalibToggle.textContent = willOpen
      ? "▾ Personal emotion calibration (advanced)"
      : "▸ Personal emotion calibration (advanced)";
  });

  // -- Personal-templates storage (opt-in localStorage). When the user opts
  // in, we serialize the resting baseline, the neutral training frames, and
  // every captured emotion template (mean + frames) so the personal classifier
  // is restored on reload. Only numerical features ever land in storage --
  // never frame data, never anything identity-bearing.
  const STORAGE_KEY_ENABLED = "expr-reader.persist";
  const STORAGE_KEY_DATA = "expr-reader.payload";
  const STORAGE_VERSION = 1;

  const isStorageEnabled = (): boolean =>
    localStorage.getItem(STORAGE_KEY_ENABLED) === "1";

  const renderStorageStatus = (): void => {
    if (!isStorageEnabled()) {
      refs.storageStatus.textContent = "(off)";
      return;
    }
    const calibrated = (Object.keys(personalTemplates) as Emotion[]).filter(
      (e) => personalTemplates[e],
    ).length;
    refs.storageStatus.textContent = `(${calibrated} template${calibrated === 1 ? "" : "s"} on this device)`;
  };

  const saveTemplatesToStorage = (): void => {
    if (!isStorageEnabled()) return;
    if (!baseline) return; // nothing useful to save without a baseline
    const payload = {
      version: STORAGE_VERSION,
      savedAt: Date.now(),
      baseline,
      neutralFrames,
      templates: personalTemplates,
    };
    try {
      localStorage.setItem(STORAGE_KEY_DATA, JSON.stringify(payload));
    } catch (err) {
      // Likely QuotaExceeded. Surface gently; don't crash the app.
      console.warn("Could not persist templates:", err);
      refs.storageStatus.textContent = "(storage full — not saved)";
    }
    renderStorageStatus();
  };

  const loadTemplatesFromStorage = (): boolean => {
    if (!isStorageEnabled()) return false;
    const raw = localStorage.getItem(STORAGE_KEY_DATA);
    if (!raw) return false;
    try {
      const payload = JSON.parse(raw);
      if (payload?.version !== STORAGE_VERSION) return false;
      if (typeof payload.baseline !== "object" || !payload.baseline) return false;
      baseline = payload.baseline as Record<string, number>;
      neutralFrames = Array.isArray(payload.neutralFrames) ? payload.neutralFrames : [];
      const stored = payload.templates ?? {};
      for (const e of EMOTIONS) {
        const t = stored[e];
        if (t && typeof t === "object" && Array.isArray(t.samples)) {
          personalTemplates[e] = t;
        }
      }
      retrainPersonalClassifier();
      renderPersonalCalibGrid();
      return true;
    } catch (err) {
      console.warn("Could not load stored templates:", err);
      return false;
    }
  };

  refs.storageEnabled.checked = isStorageEnabled();
  refs.storageEnabled.addEventListener("change", () => {
    if (refs.storageEnabled.checked) {
      localStorage.setItem(STORAGE_KEY_ENABLED, "1");
      saveTemplatesToStorage();
    } else {
      localStorage.removeItem(STORAGE_KEY_ENABLED);
      localStorage.removeItem(STORAGE_KEY_DATA);
    }
    renderStorageStatus();
  });
  refs.storageClear.addEventListener("click", () => {
    if (!window.confirm("Delete saved templates from this device? This cannot be undone.")) return;
    localStorage.removeItem(STORAGE_KEY_DATA);
    renderStorageStatus();
  });
  // Restore from disk if the user previously opted in.
  loadTemplatesFromStorage();
  renderStorageStatus();

  const onStartClick = async (): Promise<void> => {
    refs.pauseBtn.removeEventListener("click", onStartClick);
    refs.pauseBtn.disabled = true;
    refs.pauseBtn.textContent = "Loading…";
    showLoading(refs, "Loading models (~26MB)…");
    try {
      const [s, p, c] = await Promise.all([
        startWebcam(refs.video, refs.videoContainer),
        createFacePipeline(),
        createEmotionClassifier(),
      ]);
      stream = s;
      pipeline = p;
      classifier = c;
      sizeCanvasToVideo(refs.video, refs.canvas);
      hideLoading(refs);
      refs.explanation.textContent = "";
      refs.pauseBtn.textContent = "Pause";
      refs.pauseBtn.disabled = false;
      refs.pauseBtn.addEventListener("click", () => {
        paused = !paused;
        refs.pauseBtn.textContent = paused ? "Resume" : "Pause";
      });
    } catch (err) {
      hideLoading(refs);
      const msg = err instanceof Error ? err.message : String(err);
      showError(refs, `Setup failed: ${msg}`);
      refs.pauseBtn.textContent = "Retry";
      refs.pauseBtn.disabled = false;
      refs.pauseBtn.addEventListener("click", onStartClick, { once: true });
    }
  };
  refs.pauseBtn.addEventListener("click", onStartClick, { once: true });

  refs.recalibrateBtn.addEventListener("click", () => {
    startRecalibration();
  });

  refs.exportBtn.addEventListener("click", () => {
    exportSession();
  });

  // -- Build a snapshot of all current observations for Claude. Reused by
  // Why?, Summarize, and chat follow-up so every turn sees the latest state.
  const buildSnapshot = (windowSec: number = whyWindowSec): ExplainSnapshot => {
    if (!lastReadout) {
      // Caller must guard, but return a minimal valid object as a safety net.
      return {
        current: { top: [], valence: 0, arousal: 0, intensity: 0, ambiguous: false, compound: null },
        blendshapes: [],
        vaTrajectory: [],
        emotionTimeline: [],
        windowSec,
      };
    }
    const nowMs = performance.now();
    const cutoff = nowMs - windowSec * 1000;
    const windowed = timeline.filter((s) => s.t >= cutoff);

    const calibratedTemplates = (Object.keys(personalTemplates) as Emotion[])
      .filter((e) => personalTemplates[e])
      .map((e) => ({
        emotion: e,
        topBlendshapes: topTemplateBlendshapes(personalTemplates[e]!, baseline, 4),
        capturedSecondsAgo: (Date.now() - personalTemplates[e]!.capturedAt) / 1000,
      }));

    const vaTrajectory: VATrajectoryPoint[] = [];
    if (windowed.length > 0) {
      const N = Math.min(5, windowed.length);
      const tStart = windowed[0]!.t;
      for (let i = 0; i < N; i++) {
        const idx = Math.round((i / Math.max(1, N - 1)) * (windowed.length - 1));
        const sample = windowed[idx]!;
        vaTrajectory.push({
          tSec: (sample.t - tStart) / 1000,
          v: sample.valence,
          a: sample.arousal,
        });
      }
    }

    // Group consecutive same-emotion samples in the window into runs.
    const emotionTimeline: EmotionSegment[] = [];
    if (windowed.length > 0) {
      type Run = { emotion: Emotion; probSum: number; count: number; startT: number; endT: number };
      const runs: Run[] = [];
      for (const s of windowed) {
        const last = runs[runs.length - 1];
        if (last && last.emotion === s.emotion) {
          last.probSum += s.prob;
          last.count++;
          last.endT = s.t;
        } else {
          runs.push({ emotion: s.emotion, probSum: s.prob, count: 1, startT: s.t, endT: s.t });
        }
      }
      for (const r of runs) {
        emotionTimeline.push({
          startMsAgo: (nowMs - r.startT) / 1000,
          endMsAgo: (nowMs - r.endT) / 1000,
          emotion: r.emotion,
          avgProb: r.probSum / r.count,
        });
      }
    }

    let compound: string | null = null;
    if (ambiguityRun && nowMs - ambiguityRun.sinceMs >= COMPOUND_SUSTAIN_MS) {
      compound = compoundLabel(ambiguityRun.emotion1, ambiguityRun.emotion2);
    }

    const intensityNorm = baseline
      ? Math.min(1, intensityFromAdjusted(latestBlendshapes) / INTENSITY_NORM)
      : 0;

    return {
      current: {
        top: lastReadout.top,
        valence: lastReadout.valence,
        arousal: lastReadout.arousal,
        intensity: intensityNorm,
        ambiguous: lastReadout.ambiguous,
        compound,
      },
      blendshapes: topActiveBlendshapes(latestBlendshapes, 5),
      vaTrajectory,
      emotionTimeline,
      windowSec,
      personalTemplates: calibratedTemplates.length > 0 ? calibratedTemplates : undefined,
      personalNow: lastPersonalRanked
        ? lastPersonalRanked.slice(0, 3).map((p) => ({ emotion: p.emotion, prob: p.prob }))
        : undefined,
      states: lastStateScores
        ? STATES.map((st) => ({ state: st, score: lastStateScores![st] }))
        : undefined,
    };
  };

  // -- Build a session-wide snapshot for "Summarize". Uses the full session
  // timeline rather than the slider window, so Claude sees the whole arc.
  const buildSessionSnapshot = (): ExplainSnapshot => {
    if (sessionTimeline.length === 0) return buildSnapshot(whyWindowSec);
    const nowMs = performance.now();
    const sessionStart = sessionStartedAt;
    const sessionDurSec = (nowMs - sessionStart) / 1000;

    // Subsample V/A across the whole session to ~8 points.
    const vaTrajectory: VATrajectoryPoint[] = [];
    const N = Math.min(8, sessionTimeline.length);
    for (let i = 0; i < N; i++) {
      const idx = Math.round((i / Math.max(1, N - 1)) * (sessionTimeline.length - 1));
      const sample = sessionTimeline[idx]!;
      vaTrajectory.push({ tSec: sample.tSec, v: sample.valence, a: sample.arousal });
    }
    // Compress emotion runs across the entire session.
    const emotionTimeline: EmotionSegment[] = [];
    type Run = { emotion: Emotion; probSum: number; count: number; startSec: number; endSec: number };
    const runs: Run[] = [];
    for (const s of sessionTimeline) {
      const last = runs[runs.length - 1];
      if (last && last.emotion === s.emotion) {
        last.probSum += s.prob;
        last.count++;
        last.endSec = s.tSec;
      } else {
        runs.push({ emotion: s.emotion, probSum: s.prob, count: 1, startSec: s.tSec, endSec: s.tSec });
      }
    }
    for (const r of runs) {
      emotionTimeline.push({
        startMsAgo: sessionDurSec - r.startSec,
        endMsAgo: sessionDurSec - r.endSec,
        emotion: r.emotion,
        avgProb: r.probSum / r.count,
      });
    }

    const base = buildSnapshot(whyWindowSec);
    return {
      ...base,
      vaTrajectory,
      emotionTimeline,
      windowSec: sessionDurSec,
    };
  };

  // Help modal: toggle from header button, close via × / "Got it" / ESC /
  // backdrop click. Body scroll is naturally preserved by the modal's own
  // overflow-y-auto.
  const showHelp = (): void => {
    refs.helpModal.classList.remove("hidden");
    refs.helpModal.scrollTop = 0;
  };
  const hideHelp = (): void => {
    refs.helpModal.classList.add("hidden");
  };
  refs.helpToggle.addEventListener("click", showHelp);
  refs.helpClose.addEventListener("click", hideHelp);
  refs.helpCloseBottom.addEventListener("click", hideHelp);
  refs.helpModal.addEventListener("click", (e) => {
    if (e.target === refs.helpModal) hideHelp();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !refs.helpModal.classList.contains("hidden")) {
      hideHelp();
    }
  });

  // Why? window slider: range [2, 30]s, default 5s.
  refs.windowSlider.addEventListener("input", () => {
    const v = parseInt(refs.windowSlider.value, 10);
    if (!Number.isFinite(v)) return;
    whyWindowSec = v;
    refs.windowValue.textContent = `${v}s`;
  });

  // -- Recording: an opt-in capture of session samples that survives
  // Recalibrate so the user can later ask the chatbot to analyze the clip.
  // Capped at 10 minutes / 6000 samples to bound memory + payload size.
  const RECORDING_MAX_MS = 10 * 60 * 1000;
  const RECORDING_MAX_SAMPLES = 6000;
  let recording: SessionSample[] = [];
  let recordingActive = false;
  let recordingStartedAtMs = 0;

  const updateRecordButton = (): void => {
    if (recordingActive) {
      refs.recordDot.classList.remove("hidden");
      refs.recordLabel.textContent = "Stop";
      refs.recordBtn.title = "Stop recording";
    } else {
      refs.recordDot.classList.add("hidden");
      refs.recordLabel.textContent = recording.length > 0 ? "Record again" : "Record";
      refs.recordBtn.title = recording.length > 0
        ? "Recording saved — click to capture another clip (will overwrite)"
        : "Capture a clip of this session's data for later analysis (max 10 min)";
    }
    refs.discussRecording.classList.toggle("hidden", recording.length === 0);
  };

  const stopRecording = (reason: string): void => {
    if (!recordingActive) return;
    recordingActive = false;
    if (reason && recording.length > 0) {
      // Tiny system note in chat history so the user knows recording stopped.
      chatHistory.push({
        role: "assistant",
        content: `(recording stopped: ${reason}. ${recording.length} samples captured over ${((performance.now() - recordingStartedAtMs) / 1000).toFixed(1)}s. Click "Discuss recording" to analyze.)`,
      });
      renderChat();
    }
    updateRecordButton();
  };

  refs.recordBtn.addEventListener("click", () => {
    if (recordingActive) {
      stopRecording("user");
      return;
    }
    if (recording.length > 0) {
      const ok = window.confirm(
        "Replace the existing recording? The previous clip will be discarded.",
      );
      if (!ok) return;
    }
    recording = [];
    recordingActive = true;
    recordingStartedAtMs = performance.now();
    updateRecordButton();
  });
  updateRecordButton();

  // -- Chat: maintain a conversation across turns. chatHistory holds the
  // *displayed* user/assistant texts. When sending to the API we splice the
  // latest snapshot context into the most recent user message.
  // Guardrails:
  //   - min interval between Claude calls (CHAT_MIN_INTERVAL_MS)
  //   - max calls per session (CHAT_MAX_CALLS_PER_SESSION)
  //   - max chat history size sent to API (CHAT_HISTORY_CAP), oldest dropped
  //   - max user input length (CHAT_INPUT_MAX_CHARS), trimmed
  const CHAT_MIN_INTERVAL_MS = 1500;
  const CHAT_MAX_CALLS_PER_SESSION = 40;
  const CHAT_HISTORY_CAP = 30;
  const CHAT_INPUT_MAX_CHARS = 600;
  const chatHistory: ChatMessage[] = [];
  let chatCallCount = 0;
  let lastChatCallAt = 0;
  let activeAssistantEl: HTMLElement | null = null;

  const renderChat = (): void => {
    refs.explanation.innerHTML = "";
    activeAssistantEl = null;
    for (const msg of chatHistory) {
      const wrap = document.createElement("div");
      wrap.className = "mb-3";
      const tag = document.createElement("div");
      tag.className =
        "text-[10px] font-mono uppercase tracking-widest mb-0.5 " +
        (msg.role === "user" ? "text-indigo-400" : "text-neutral-500");
      tag.textContent = msg.role === "user" ? "you" : "assistant";
      const body = document.createElement("div");
      body.className =
        "whitespace-pre-wrap text-sm " +
        (msg.role === "user" ? "text-neutral-300" : "text-neutral-100");
      body.textContent = msg.content;
      wrap.appendChild(tag);
      wrap.appendChild(body);
      refs.explanation.appendChild(wrap);
    }
    if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1]!.role === "assistant") {
      const lastBody = refs.explanation.lastChild?.lastChild as HTMLElement | null;
      activeAssistantEl = lastBody ?? null;
    }
    refs.explanation.scrollTop = refs.explanation.scrollHeight;
  };

  type SnapshotMode = "window" | "session" | "recording";

  const buildRecordingSnapshot = (): ExplainSnapshot => {
    const base = buildSnapshot(whyWindowSec);
    if (recording.length === 0) return base;
    const durSec = recording[recording.length - 1]!.tSec - recording[0]!.tSec;
    // Subsample V/A to ~10 points across the recording.
    const vaTrajectory: VATrajectoryPoint[] = [];
    const N = Math.min(10, recording.length);
    const tStart = recording[0]!.tSec;
    for (let i = 0; i < N; i++) {
      const idx = Math.round((i / Math.max(1, N - 1)) * (recording.length - 1));
      const sample = recording[idx]!;
      vaTrajectory.push({
        tSec: sample.tSec - tStart,
        v: sample.valence,
        a: sample.arousal,
      });
    }
    const emotionTimeline: EmotionSegment[] = [];
    type Run = { emotion: Emotion; probSum: number; count: number; startSec: number; endSec: number };
    const runs: Run[] = [];
    for (const s of recording) {
      const last = runs[runs.length - 1];
      if (last && last.emotion === s.emotion) {
        last.probSum += s.prob;
        last.count++;
        last.endSec = s.tSec;
      } else {
        runs.push({ emotion: s.emotion, probSum: s.prob, count: 1, startSec: s.tSec, endSec: s.tSec });
      }
    }
    const recDurSec = durSec;
    for (const r of runs) {
      emotionTimeline.push({
        startMsAgo: recDurSec - (r.startSec - tStart),
        endMsAgo: recDurSec - (r.endSec - tStart),
        emotion: r.emotion,
        avgProb: r.probSum / r.count,
      });
    }
    return { ...base, vaTrajectory, emotionTimeline, windowSec: durSec };
  };

  const sendChatTurn = async (userText: string, mode: SnapshotMode): Promise<void> => {
    if (!lastReadout && mode !== "recording") {
      refs.explanation.classList.add("text-amber-300");
      refs.explanation.textContent = "No face is being tracked yet.";
      return;
    }
    if (mode === "recording" && recording.length === 0) {
      return;
    }
    const trimmed = userText.length > CHAT_INPUT_MAX_CHARS
      ? userText.slice(0, CHAT_INPUT_MAX_CHARS) + "…"
      : userText;
    const nowMs = performance.now();
    if (nowMs - lastChatCallAt < CHAT_MIN_INTERVAL_MS) {
      return; // silently rate-limit; UI keeps the buttons disabled briefly anyway
    }
    if (chatCallCount >= CHAT_MAX_CALLS_PER_SESSION) {
      chatHistory.push({
        role: "assistant",
        content: `(rate limit: this session has used ${CHAT_MAX_CALLS_PER_SESSION} chat calls — refresh to reset)`,
      });
      renderChat();
      return;
    }
    refs.explanation.classList.remove("text-red-300", "text-amber-300");
    if (explainAbort) {
      explainAbort.abort();
      explainAbort = null;
    }

    chatHistory.push({ role: "user", content: trimmed });
    const assistantMsg: ChatMessage = { role: "assistant", content: "" };
    chatHistory.push(assistantMsg);
    renderChat();

    const snapshot =
      mode === "session" ? buildSessionSnapshot()
      : mode === "recording" ? buildRecordingSnapshot()
      : buildSnapshot(whyWindowSec);

    // Cap history sent to API; always preserve the very first turn (often a
    // summary that anchors the conversation).
    const apiMessages: ChatMessage[] = chatHistory.map((m) => ({ ...m }));
    apiMessages.pop(); // drop placeholder assistant
    if (apiMessages.length > CHAT_HISTORY_CAP) {
      const head = apiMessages.slice(0, 1);
      const tail = apiMessages.slice(-(CHAT_HISTORY_CAP - 1));
      apiMessages.length = 0;
      apiMessages.push(...head, ...tail);
    }
    apiMessages[apiMessages.length - 1] = {
      role: "user",
      content: `${buildPrompt(snapshot)}\n\nUser question: ${trimmed}`,
    };

    explainAbort = new AbortController();
    chatCallCount++;
    lastChatCallAt = nowMs;
    refs.chatSend.disabled = true;
    refs.summarizeSession.disabled = true;
    refs.explainBtn.disabled = true;
    refs.discussRecording.disabled = true;
    try {
      for await (const chunk of streamChat(apiMessages, explainAbort.signal)) {
        assistantMsg.content += chunk;
        if (activeAssistantEl) {
          activeAssistantEl.textContent = assistantMsg.content;
          refs.explanation.scrollTop = refs.explanation.scrollHeight;
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const errMsg = err instanceof Error ? err.message : String(err);
      assistantMsg.content = `(error: ${errMsg})`;
      if (activeAssistantEl) activeAssistantEl.textContent = assistantMsg.content;
    } finally {
      refs.chatSend.disabled = false;
      refs.summarizeSession.disabled = false;
      refs.explainBtn.disabled = false;
      refs.discussRecording.disabled = false;
      explainAbort = null;
    }
  };

  refs.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = refs.chatInput.value.trim();
    if (!text) return;
    refs.chatInput.value = "";
    void sendChatTurn(text, "window");
  });

  refs.summarizeSession.addEventListener("click", () => {
    void sendChatTurn(
      "Summarize my emotional journey across the entire session — when I shifted, what muscles drove it, and what stood out.",
      "session",
    );
  });

  refs.discussRecording.addEventListener("click", () => {
    void sendChatTurn(
      "Analyze the saved recording — describe the arc, key transitions, dominant muscle patterns, and any contradictions in the signals.",
      "recording",
    );
  });

  refs.explainBtn.addEventListener("click", () => {
    void sendChatTurn(
      "Summarize what's been happening over this window — emotion, movement, and which muscles drove it.",
      "window",
    );
  });

  const tick = (): void => {
    if (!paused && stream && pipeline && classifier && refs.video.readyState >= 2) {
      sizeCanvasToVideo(refs.video, refs.canvas);
      const now = performance.now();
      const ts = now > lastTimestamp ? now : lastTimestamp + 1;
      lastTimestamp = ts;

      const faces = pipeline.process(refs.video, ts);

      // Run emotion classification on the primary face, async, non-blocking.
      // Multi-face emotion is a v2 lever -- single classifier inference per
      // frame keeps us at 30fps on a mid-range laptop.
      if (faces.length > 0 && !inferenceInFlight) {
        inferenceInFlight = true;
        const face = faces[0]!;
        // Snapshot the blendshapes for this frame so the personal classifier
        // sees the same input the HSEmotion crop did, even if the face moves
        // by the time inference resolves.
        const rawBsForFrame = face.blendshapes;
        classifier
          .classify(refs.video, face.bbox)
          .then((r) => {
            lastReadout = toDisplay(r);
            // Track sustained ambiguity for compound-emotion labeling.
            if (lastReadout.ambiguous && lastReadout.top.length >= 2) {
              const e1 = lastReadout.top[0]!.emotion;
              const e2 = lastReadout.top[1]!.emotion;
              const key = e1 < e2 ? `${e1}|${e2}` : `${e2}|${e1}`;
              if (!ambiguityRun || ambiguityRun.pair !== key) {
                ambiguityRun = { pair: key, emotion1: e1, emotion2: e2, sinceMs: performance.now() };
              }
            } else {
              ambiguityRun = null;
            }
            probHistory.push(lastReadout.top[0]!.prob);
            if (probHistory.length > SPARKLINE_FRAMES) probHistory.shift();
            const tNow = performance.now();
            vaTrail.push({ v: r.valence, a: r.arousal, t: tNow });
            const vaCutoff = tNow - VA_TRAIL_MS;
            while (vaTrail.length > 0 && vaTrail[0]!.t < vaCutoff) vaTrail.shift();
            timeline.push({
              t: tNow,
              emotion: lastReadout.top[0]!.emotion,
              prob: lastReadout.top[0]!.prob,
              valence: r.valence,
              arousal: r.arousal,
            });
            const tlCutoff = tNow - TIMELINE_MS;
            while (timeline.length > 0 && timeline[0]!.t < tlCutoff) timeline.shift();

            // Personal classifier (diagonal-Gaussian, trained from the user's
            // calibration snapshots). Runs in lockstep with HSEmotion so the
            // two probabilities are computed against the same blendshape frame.
            lastPersonalRanked = personalClassifier
              ? personalClassifier.predict(rawBsForFrame)
              : null;

            // Append to the session timeline. Captures the state at this
            // inference for later JSON export. Adjusted blendshapes (after
            // baseline subtraction) are used so exports represent meaningful
            // muscle deltas, not the user's neutral asymmetry.
            const adjustedFrame = applyBaseline(rawBsForFrame);
            const intensity = intensityFromAdjusted(adjustedFrame);
            const rawEyeBlinkNow = (
              (rawBsForFrame["eyeBlinkLeft"] ?? 0) +
              (rawBsForFrame["eyeBlinkRight"] ?? 0)
            ) * 0.5;
            if (sessionStartedAt === 0) sessionStartedAt = tNow;
            // Capture personal-baseline samples during the 3 s neutral
            // calibration so we can anchor state thresholds to this user.
            if (!baseline) {
              baselineSamples.push({
                eyeBlink: rawEyeBlinkNow,
                intensity,
                valence: r.valence,
                arousal: r.arousal,
              });
              baselineSampleTimes.push(tNow);
            }

            const sample: SessionSample = {
              tSec: (tNow - sessionStartedAt) / 1000,
              emotion: lastReadout.top[0]!.emotion,
              prob: lastReadout.top[0]!.prob,
              valence: r.valence,
              arousal: r.arousal,
              intensity,
              topBlendshapes: topActiveBlendshapes(adjustedFrame, 5),
            };
            sessionTimeline.push(sample);
            if (sessionTimeline.length > SESSION_MAX_SAMPLES) sessionTimeline.shift();

            // Cognitive states. Push a fresh sample into the 60s long window
            // and estimate samplesPerSec from the actual timestamps. PERCLOS
            // uses RAW eyeBlink (not baseline-subtracted) since the literature
            // threshold is on absolute eye closure, not deviation from rest.
            longWindow.push({
              eyeBlink: rawEyeBlinkNow,
              intensity,
              valence: r.valence,
              arousal: r.arousal,
            });
            longWindowTimes.push(tNow);
            if (longWindow.length > LONG_WINDOW_SAMPLES) longWindow.shift();
            if (longWindowTimes.length > LONG_WINDOW_SAMPLES) longWindowTimes.shift();
            const samplesPerSec =
              longWindowTimes.length >= 2
                ? (longWindowTimes.length - 1) /
                  Math.max(0.1, (longWindowTimes[longWindowTimes.length - 1]! - longWindowTimes[0]!) / 1000)
                : 10;
            const intensityNormForStates = baseline
              ? Math.min(1, intensity / INTENSITY_NORM)
              : 0;
            // Don't compute states until the long window has at least 60s of
            // data. PERCLOS, blink rate, and intensity variance all need a
            // minute-class window to be meaningful per the literature; below
            // that threshold the readings would be both noisy and not
            // technically PERCLOS at all.
            if (longWindow.length >= STATES_MIN_SAMPLES) {
              const rawStateScores = computeStateScores({
                blendshapes: adjustedFrame,
                rawBlendshapes: rawBsForFrame,
                valence: r.valence,
                arousal: r.arousal,
                intensityNorm: intensityNormForStates,
                ambiguous: lastReadout.ambiguous,
                longWindow,
                samplesPerSec,
                userBaseline: userBaseline ?? undefined,
              });
              if (!smoothedStateScores) {
                smoothedStateScores = { ...rawStateScores };
              } else {
                const a = STATE_SMOOTHING_ALPHA;
                for (const s of STATES) {
                  smoothedStateScores[s] =
                    (1 - a) * smoothedStateScores[s] + a * rawStateScores[s];
                }
              }
              lastStateScores = smoothedStateScores;
            } else {
              lastStateScores = null;
            }

            // Recording: opt-in clip with hard caps on duration and sample
            // count so we never accumulate unbounded data.
            if (recordingActive) {
              recording.push(sample);
              const elapsedMs = tNow - recordingStartedAtMs;
              if (
                recording.length >= RECORDING_MAX_SAMPLES ||
                elapsedMs >= RECORDING_MAX_MS
              ) {
                stopRecording("max length reached");
              }
            }

            // Auto-trigger evaluation for the Why? pulse. Three reasons:
            // sustained ambiguity, big V/A jump, or HSEmotion vs personal
            // classifier disagreement. We re-arm only after a cooldown so the
            // bbox doesn't constantly pulse during a long ambiguous stretch.
            const reasons: string[] = [];
            if (ambiguityRun && tNow - ambiguityRun.sinceMs > AMBIGUITY_INTERESTING_MS) {
              reasons.push("ambiguity > 3s");
            }
            if (prevVA) {
              const dv = r.valence - prevVA.v;
              const da = r.arousal - prevVA.a;
              if (Math.sqrt(dv * dv + da * da) > VA_JUMP_THRESHOLD) {
                reasons.push("affect jump");
              }
            }
            if (
              lastPersonalRanked &&
              lastPersonalRanked.length > 0 &&
              lastPersonalRanked[0]!.emotion !== lastReadout.top[0]!.emotion
            ) {
              reasons.push("classifier disagreement");
            }
            prevVA = { v: r.valence, a: r.arousal };
            if (reasons.length > 0 && tNow > pulseUntilMs && tNow > pulseCooldownUntilMs) {
              pulseUntilMs = tNow + PULSE_DURATION_MS;
              pulseCooldownUntilMs = pulseUntilMs + PULSE_COOLDOWN_MS;
              pulseReason = reasons[0]!;
            }
          })
          .catch((err) => {
            console.error("classify failed", err);
          })
          .finally(() => {
            inferenceInFlight = false;
          });
      } else if (faces.length === 0) {
        // No face means no current subject; don't bleed the prior person's
        // history into whoever shows up next. Baseline persists -- recapturing
        // it on every micro-occlusion would be more annoying than helpful;
        // user can click Recalibrate.
        probHistory.length = 0;
        vaTrail.length = 0;
        timeline.length = 0;
        latestBlendshapes = {};
        lastReadout = null;
        lastStateScores = null;
        smoothedStateScores = null;
        longWindow.length = 0;
        longWindowTimes.length = 0;
        // Pulse state: clear so the bbox doesn't reappear with a stale "?".
        pulseUntilMs = 0;
        pulseReason = null;
        prevVA = null;
      }

      if (faces.length > 0) {
        const rawBs = faces[0]!.blendshapes;
        if (!baseline) {
          // Calibration window: accumulate per-frame snapshots so we can both
          // compute the resting-face baseline AND keep them as the personal
          // classifier's "neutral" training class.
          if (calibStartedAt === 0) {
            calibStartedAt = performance.now();
            showLoading(refs, "Calibrating: hold a neutral expression…");
          }
          calibFrames.push({ ...rawBs });
          if (performance.now() - calibStartedAt >= CALIB_DURATION_MS) {
            baseline = meanFrame(calibFrames);
            neutralFrames = calibFrames;
            calibFrames = [];
            // Compute the user's personal resting metrics from samples
            // captured in this calibration window. Used to anchor PERCLOS,
            // blink-rate, and expression-variance thresholds in states.ts.
            if (baselineSamples.length >= 3) {
              const span =
                (baselineSampleTimes[baselineSampleTimes.length - 1]! -
                  baselineSampleTimes[0]!) / 1000;
              const sps = (baselineSamples.length - 1) / Math.max(0.5, span);
              userBaseline = computeUserBaseline(baselineSamples, sps);
            }
            hideLoading(refs);
            retrainPersonalClassifier();
            saveTemplatesToStorage();
          }
        } else if (personalCalibState) {
          // Personal-emotion capture: keep every frame so the classifier sees
          // the user's intra-window variance, not just the mean. With one
          // sample per class an LR/NB classifier would over-fit; a 3s × ~10Hz
          // capture gives ~30 samples per class.
          if (personalCalibState.startedAt === 0) {
            personalCalibState.startedAt = performance.now();
            showPersonalCalib(refs, personalCalibState.emotion);
          }
          personalCalibState.frames.push({ ...rawBs });
          const elapsedMs = performance.now() - personalCalibState.startedAt;
          const remaining = Math.max(0, Math.ceil((CALIB_DURATION_MS - elapsedMs) / 1000));
          updatePersonalCalib(refs, remaining, elapsedMs / CALIB_DURATION_MS);
          if (elapsedMs >= CALIB_DURATION_MS) {
            personalTemplates[personalCalibState.emotion] = {
              blendshapes: meanFrame(personalCalibState.frames),
              samples: personalCalibState.frames,
              capturedAt: Date.now(),
            };
            personalCalibState = null;
            hidePersonalCalib(refs);
            renderPersonalCalibGrid();
            retrainPersonalClassifier();
            saveTemplatesToStorage();
          }
        }
        latestBlendshapes = applyBaseline(rawBs);
      }

      refs.ctx.clearRect(0, 0, refs.canvas.width, refs.canvas.height);
      const inAnyCalibration = !baseline || personalCalibState !== null;
      if (!inAnyCalibration) {
        // Compound label only fires when ambiguity has held the same top-2
        // pair for ≥COMPOUND_SUSTAIN_MS AND the pair is in the DTM14 lookup.
        let compound: string | null = null;
        if (ambiguityRun && performance.now() - ambiguityRun.sinceMs >= COMPOUND_SUSTAIN_MS) {
          compound = compoundLabel(ambiguityRun.emotion1, ambiguityRun.emotion2);
        }
        const tNow = performance.now();
        const pulseActive = tNow < pulseUntilMs;
        const intensityNorm = baseline
          ? Math.min(1, intensityFromAdjusted(latestBlendshapes) / INTENSITY_NORM)
          : 0;
        // Warmup status for the states block: while the long window fills
        // toward 60s, show "warming up Xs / 60s" instead of bogus scores.
        const stateWarmup =
          lastStateScores === null && longWindow.length > 0
            ? {
                samples: longWindow.length,
                target: STATES_MIN_SAMPLES,
                seconds: Math.floor(longWindow.length / Math.max(1, samplesPerSecLive())),
              }
            : null;

        // Real-time eye tracking & gaze targeting on the document reader
        if (faces.length > 0) {
          const primaryFace = faces[0]!;
          lastGazeResult = gazeTracker.process(
            primaryFace,
            docChat.getReaderElement(),
            refs.videoContainer,
            tNow,
          );

          // Update facial comprehension & confusion feedback for Gemma
          const confusionScore = lastStateScores ? (lastStateScores["confused"] ?? 0) : 0;
          const stressedScore = lastStateScores ? (lastStateScores["stressed"] ?? 0) : 0;
          const focusedScore = lastStateScores ? (lastStateScores["focused"] ?? 0) : 0.5;
          const topActive = topActiveBlendshapes(latestBlendshapes, 3).map((b) => `${b.name}: ${b.score.toFixed(2)}`);

          docChat.updateComprehension(lastGazeResult, {
            topEmotion: lastReadout?.top[0]?.emotion ?? "neutral",
            emotionConfidence: lastReadout?.top[0]?.prob ?? 0.8,
            cognitiveState: lastStateScores ? (rankStates(lastStateScores, 1)[0]?.state ?? "focused") : "focused",
            confusionScore,
            stressedScore,
            focusedScore,
            valence: lastReadout?.valence ?? 0,
            arousal: lastReadout?.arousal ?? 0,
            activeMuscles: topActive,
          });
        }

        // Update large feature dashboard displayed below the camera
        dashboard.update(
          faces.length > 0 ? lastReadout : null,
          compound,
          lastGazeResult,
          intensityNorm,
          lastStateScores,
          stateWarmup,
          topActiveBlendshapes(latestBlendshapes, 3),
          lastPersonalRanked,
        );

        drawDebugBoxes(
          refs.ctx,
          faces,
          faces.length > 0 ? lastReadout : null,
          probHistory,
          latestBlendshapes,
          lastPersonalRanked,
          compound,
          pulseActive ? pulseReason : null,
          tNow,
          intensityNorm,
          lastStateScores,
          stateWarmup,
        );
        if (faces.length > 0 && lastReadout) {
          const insetSize = 140;
          const insetMargin = 16;
          const ix = refs.canvas.width - insetSize - insetMargin;
          const iy = refs.canvas.height - insetSize - insetMargin;
          drawValenceArousal(refs.ctx, ix, iy, insetSize, vaTrail, performance.now());
        }
      } else {
        // Calibration phase: just stroke the bbox so the user knows we see them.
        refs.ctx.lineWidth = 3;
        refs.ctx.strokeStyle = "rgba(99, 102, 241, 0.55)";
        for (const f of faces) {
          refs.ctx.strokeRect(f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h);
        }
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

main().catch((err) => {
  console.error(err);
});

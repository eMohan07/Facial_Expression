// Feature Dashboard below the camera
// Displays large, high-visibility cards for emotions, gaze radar, cognitive states,
// valence/arousal, and muscle activations.

import type { Emotion } from "../emotion-head";
import type { CognitiveState } from "../states";
import type { GazeResult } from "../gaze-tracker";
import type { PersonalProb } from "../personal-classifier";

export interface DisplayReadout {
  top: { emotion: Emotion; prob: number }[];
  valence: number;
  arousal: number;
  ambiguous: boolean;
}

const EMOTION_ICONS: Record<Emotion, string> = {
  happy: "😊",
  neutral: "😐",
  surprised: "😮",
  sad: "😢",
  angry: "😠",
  fear: "😨",
  disgust: "🤢",
  contempt: "😏",
};

const EMOTION_COLORS: Record<Emotion, string> = {
  happy: "bg-emerald-500",
  neutral: "bg-neutral-400",
  surprised: "bg-sky-400",
  sad: "bg-indigo-400",
  angry: "bg-rose-500",
  fear: "bg-amber-500",
  disgust: "bg-teal-500",
  contempt: "bg-purple-400",
};

const STATE_ICONS: Record<CognitiveState, string> = {
  focused: "🎯",
  engaged: "⚡",
  calm: "🧘",
  tired: "😴",
  bored: "🥱",
  stressed: "🤯",
  confused: "❓",
};

export class FeatureDashboard {
  // Primary Emotion elements
  private primaryIconEl: HTMLElement;
  private primaryNameEl: HTMLElement;
  private primaryPctEl: HTMLElement;
  private compoundBadgeEl: HTMLElement;
  private emotionBarsContainer: HTMLElement;

  // Gaze & Attention elements
  private gazeStatusEl: HTMLElement;
  private gazeTargetEl: HTMLElement;
  private gazeAttentionBarEl: HTMLElement;
  private gazeAttentionPctEl: HTMLElement;
  private gazeCoordsEl: HTMLElement;
  private eyeOpenLeftEl: HTMLElement;
  private eyeOpenRightEl: HTMLElement;

  // Cognitive States elements
  private statesGridEl: HTMLElement;
  private stateWarmupNoticeEl: HTMLElement;

  // Valence / Arousal elements
  private valenceValEl: HTMLElement;
  private valenceBarEl: HTMLElement;
  private arousalValEl: HTMLElement;
  private arousalBarEl: HTMLElement;

  // Intensity elements
  private intensityValEl: HTMLElement;
  private intensityBarEl: HTMLElement;

  // Blendshapes elements
  private blendshapesListEl: HTMLElement;

  constructor() {
    this.primaryIconEl = document.getElementById("dash-emotion-icon")!;
    this.primaryNameEl = document.getElementById("dash-emotion-name")!;
    this.primaryPctEl = document.getElementById("dash-emotion-pct")!;
    this.compoundBadgeEl = document.getElementById("dash-compound-badge")!;
    this.emotionBarsContainer = document.getElementById("dash-emotion-bars")!;

    this.gazeStatusEl = document.getElementById("dash-gaze-status")!;
    this.gazeTargetEl = document.getElementById("dash-gaze-target")!;
    this.gazeAttentionBarEl = document.getElementById("dash-gaze-attn-bar")!;
    this.gazeAttentionPctEl = document.getElementById("dash-gaze-attn-pct")!;
    this.gazeCoordsEl = document.getElementById("dash-gaze-coords")!;
    this.eyeOpenLeftEl = document.getElementById("dash-eye-left")!;
    this.eyeOpenRightEl = document.getElementById("dash-eye-right")!;

    this.statesGridEl = document.getElementById("dash-states-grid")!;
    this.stateWarmupNoticeEl = document.getElementById("dash-state-warmup")!;

    this.valenceValEl = document.getElementById("dash-valence-val")!;
    this.valenceBarEl = document.getElementById("dash-valence-bar")!;
    this.arousalValEl = document.getElementById("dash-arousal-val")!;
    this.arousalBarEl = document.getElementById("dash-arousal-bar")!;

    this.intensityValEl = document.getElementById("dash-intensity-val")!;
    this.intensityBarEl = document.getElementById("dash-intensity-bar")!;

    this.blendshapesListEl = document.getElementById("dash-blendshapes-list")!;
  }

  /**
   * Update full features dashboard with latest inference frame data
   */
  public update(
    readout: DisplayReadout | null,
    compound: string | null,
    gaze: GazeResult | null,
    intensityNorm: number,
    stateScores: Record<CognitiveState, number> | null,
    stateWarmup: { seconds: number; target: number } | null,
    topBlendshapes: ReadonlyArray<{ name: string; score: number }>,
    personalRanked: PersonalProb[] | null,
  ): void {
    if (readout) {
      this.updateEmotion(readout, compound, personalRanked);
      this.updateValenceArousal(readout.valence, readout.arousal);
    }
    if (gaze) {
      this.updateGaze(gaze);
    }
    this.updateIntensity(intensityNorm);
    this.updateCognitiveStates(stateScores, stateWarmup);
    this.updateBlendshapes(topBlendshapes);
  }

  private updateEmotion(
    r: DisplayReadout,
    compound: string | null,
    _personalRanked: PersonalProb[] | null,
  ): void {
    if (!r.top || r.top.length === 0) return;
    const top = r.top[0]!;
    const icon = EMOTION_ICONS[top.emotion] ?? "🙂";

    if (this.primaryIconEl) this.primaryIconEl.textContent = icon;
    if (this.primaryNameEl) this.primaryNameEl.textContent = top.emotion.toUpperCase();
    if (this.primaryPctEl) this.primaryPctEl.textContent = `${(top.prob * 100).toFixed(0)}%`;

    // Compound badge
    if (this.compoundBadgeEl) {
      if (compound) {
        this.compoundBadgeEl.textContent = `✨ ${compound.toUpperCase()}`;
        this.compoundBadgeEl.classList.remove("hidden");
      } else if (r.ambiguous) {
        this.compoundBadgeEl.textContent = "⚡ AMBIGUOUS (TOP-2 CLOSE)";
        this.compoundBadgeEl.classList.remove("hidden");
      } else {
        this.compoundBadgeEl.classList.add("hidden");
      }
    }

    // Emotion bars
    if (this.emotionBarsContainer) {
      this.emotionBarsContainer.innerHTML = "";
      r.top.forEach((item, idx) => {
        const isTop = idx === 0;
        const barColor = EMOTION_COLORS[item.emotion] ?? "bg-indigo-500";
        const pct = (item.prob * 100).toFixed(0);

        const row = document.createElement("div");
        row.className = "flex items-center gap-2.5 text-xs";

        const label = document.createElement("span");
        label.className = `w-24 font-medium capitalize truncate ${isTop ? "text-white font-bold" : "text-neutral-400"}`;
        label.textContent = `${EMOTION_ICONS[item.emotion]} ${item.emotion}`;

        const barTrack = document.createElement("div");
        barTrack.className = "flex-1 h-3 bg-neutral-800 rounded-full overflow-hidden";

        const barFill = document.createElement("div");
        barFill.className = `h-full ${barColor} rounded-full transition-all duration-100 ease-out`;
        barFill.style.width = `${pct}%`;
        barTrack.appendChild(barFill);

        const pctSpan = document.createElement("span");
        pctSpan.className = `w-9 text-right font-mono tabular-nums ${isTop ? "text-indigo-300 font-bold" : "text-neutral-400"}`;
        pctSpan.textContent = `${pct}%`;

        row.appendChild(label);
        row.appendChild(barTrack);
        row.appendChild(pctSpan);
        this.emotionBarsContainer.appendChild(row);
      });
    }
  }

  private updateGaze(gaze: GazeResult): void {
    if (this.gazeStatusEl) {
      this.gazeStatusEl.textContent = gaze.statusText;
      if (gaze.isOnNotepad) {
        this.gazeStatusEl.className = "font-bold text-emerald-400 text-sm";
      } else if (gaze.lookingRegion === "camera") {
        this.gazeStatusEl.className = "font-bold text-indigo-400 text-sm";
      } else {
        this.gazeStatusEl.className = "font-bold text-neutral-300 text-sm";
      }
    }

    if (this.gazeTargetEl) {
      if (gaze.isOnNotepad) {
        this.gazeTargetEl.textContent = `📍 Notepad Line ${gaze.targetLineIndex + 1}`;
      } else {
        this.gazeTargetEl.textContent = `👀 ${gaze.lookingRegion.toUpperCase()}`;
      }
    }

    if (this.gazeAttentionBarEl && this.gazeAttentionPctEl) {
      this.gazeAttentionBarEl.style.width = `${gaze.attentionScore}%`;
      this.gazeAttentionPctEl.textContent = `${gaze.attentionScore}%`;
    }

    if (this.gazeCoordsEl) {
      const xPct = Math.round(((gaze.normX + 1) / 2) * 100);
      const yPct = Math.round(((gaze.normY + 1) / 2) * 100);
      this.gazeCoordsEl.textContent = `X: ${xPct}% · Y: ${yPct}% (${gaze.horizontalDirection}/${gaze.verticalDirection})`;
    }

    if (this.eyeOpenLeftEl) {
      const lPct = Math.round(gaze.leftEyeOpen * 100);
      this.eyeOpenLeftEl.textContent = `L: ${lPct}%`;
    }

    if (this.eyeOpenRightEl) {
      const rPct = Math.round(gaze.rightEyeOpen * 100);
      this.eyeOpenRightEl.textContent = `R: ${rPct}%`;
    }
  }

  private updateCognitiveStates(
    scores: Record<CognitiveState, number> | null,
    warmup: { seconds: number; target: number } | null,
  ): void {
    if (!this.statesGridEl) return;

    if (warmup) {
      if (this.stateWarmupNoticeEl) {
        this.stateWarmupNoticeEl.classList.remove("hidden");
        const fill = document.getElementById("dash-warmup-bar");
        const text = document.getElementById("dash-warmup-text");
        const pct = Math.min(100, Math.round((warmup.seconds / 60) * 100));
        if (fill) fill.style.width = `${pct}%`;
        if (text) text.textContent = `Warming up literature window: ${warmup.seconds}s / 60s`;
      }
    } else {
      if (this.stateWarmupNoticeEl) {
        this.stateWarmupNoticeEl.classList.add("hidden");
      }
    }

    if (scores) {
      const statesList: CognitiveState[] = [
        "focused",
        "engaged",
        "calm",
        "tired",
        "bored",
        "stressed",
        "confused",
      ];

      // Find top state
      let topState: CognitiveState = "focused";
      let topVal = -1;
      statesList.forEach((st) => {
        const val = scores[st] ?? 0;
        if (val > topVal) {
          topVal = val;
          topState = st;
        }
      });

      this.statesGridEl.innerHTML = "";
      statesList.forEach((st) => {
        const val = scores[st] ?? 0;
        const isTop = st === topState && topVal > 0.35;
        const pct = Math.round(val * 100);

        const card = document.createElement("div");
        card.className = `p-2 rounded-lg border text-xs transition-all ${
          isTop
            ? "bg-indigo-600/20 border-indigo-500 shadow-md ring-1 ring-indigo-500/40"
            : "bg-neutral-900/60 border-neutral-800 text-neutral-400"
        }`;

        const head = document.createElement("div");
        head.className = "flex items-center justify-between mb-1";

        const title = document.createElement("span");
        title.className = `font-medium capitalize ${isTop ? "text-indigo-200 font-bold" : "text-neutral-300"}`;
        title.textContent = `${STATE_ICONS[st]} ${st}`;

        const scoreSpan = document.createElement("span");
        scoreSpan.className = `font-mono tabular-nums ${isTop ? "text-indigo-300 font-bold" : "text-neutral-500"}`;
        scoreSpan.textContent = `${pct}%`;

        head.appendChild(title);
        head.appendChild(scoreSpan);

        const barTrack = document.createElement("div");
        barTrack.className = "w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden";

        const barFill = document.createElement("div");
        barFill.className = `h-full rounded-full ${isTop ? "bg-indigo-400" : "bg-neutral-600"}`;
        barFill.style.width = `${pct}%`;
        barTrack.appendChild(barFill);

        card.appendChild(head);
        card.appendChild(barTrack);
        this.statesGridEl.appendChild(card);
      });
    }
  }

  private updateValenceArousal(valence: number, arousal: number): void {
    if (this.valenceValEl) {
      const sign = valence > 0 ? "+" : "";
      this.valenceValEl.textContent = `${sign}${valence.toFixed(2)}`;
      this.valenceValEl.className = `font-mono font-bold text-sm ${
        valence > 0.15 ? "text-emerald-400" : valence < -0.15 ? "text-rose-400" : "text-neutral-300"
      }`;
    }

    if (this.valenceBarEl) {
      const pct = Math.round(((valence + 1) / 2) * 100);
      this.valenceBarEl.style.width = `${pct}%`;
    }

    if (this.arousalValEl) {
      const sign = arousal > 0 ? "+" : "";
      this.arousalValEl.textContent = `${sign}${arousal.toFixed(2)}`;
      this.arousalValEl.className = `font-mono font-bold text-sm ${
        arousal > 0.15 ? "text-amber-400" : arousal < -0.15 ? "text-sky-400" : "text-neutral-300"
      }`;
    }

    if (this.arousalBarEl) {
      const pct = Math.round(((arousal + 1) / 2) * 100);
      this.arousalBarEl.style.width = `${pct}%`;
    }
  }

  private updateIntensity(intensityNorm: number): void {
    if (this.intensityValEl) {
      this.intensityValEl.textContent = `${Math.round(intensityNorm * 100)}%`;
    }
    if (this.intensityBarEl) {
      this.intensityBarEl.style.width = `${Math.round(intensityNorm * 100)}%`;
    }
  }

  private updateBlendshapes(topBs: ReadonlyArray<{ name: string; score: number }>): void {
    if (!this.blendshapesListEl) return;
    this.blendshapesListEl.innerHTML = "";
    if (topBs.length === 0) {
      const empty = document.createElement("span");
      empty.className = "text-neutral-500 text-xs italic";
      empty.textContent = "Neutral / Resting";
      this.blendshapesListEl.appendChild(empty);
      return;
    }

    topBs.forEach((bs) => {
      const chip = document.createElement("div");
      chip.className =
        "flex items-center justify-between px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-xs";
      const name = document.createElement("span");
      name.className = "text-neutral-300 font-mono text-[11px] truncate max-w-[150px]";
      name.textContent = bs.name;

      const val = document.createElement("span");
      val.className = "text-indigo-400 font-mono font-semibold tabular-nums ml-2";
      val.textContent = bs.score.toFixed(2);

      chip.appendChild(name);
      chip.appendChild(val);
      this.blendshapesListEl.appendChild(chip);
    });
  }
}

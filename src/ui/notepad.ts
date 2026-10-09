// Interactive Notepad with Real-Time Gaze Reading Tracking
// Supports word insertion, clipboard copy/paste, sample passages, font sizing,
// and gaze-driven line highlighting with floating reticle.

import type { GazeResult, GazeTracker } from "../gaze-tracker";

const SAMPLE_PASSAGES = [
  {
    title: "The Science of Visual Attention",
    text: `Visual attention is the cognitive mechanism through which our brain selectively concentrates on discrete aspects of sensory information while ignoring extraneous background noise.

When you read a line of text, your eyes do not glide smoothly along the letters. Instead, they make rapid, microscopic jumps called saccades, pausing momentarily on specific words during events called fixations.

During each fixation, which typically lasts between 200 and 300 milliseconds, your visual cortex extracts phonetic, morphological, and semantic information from the text.

Eye-tracking technology captures these subtle ocular movements in real time. By analyzing gaze dwell times and pupil trajectories, we can accurately measure reading engagement, cognitive workload, and comprehension depth.`,
  },
  {
    title: "Real-Time Facial Expression & Emotion",
    text: `Human emotions manifest across subtle micro-movements of facial musculature described by the Facial Action Coding System (FACS).

The zygomaticus major muscle draws the lip corners obliquely upward into a genuine smile, while the orbicularis oculi tightens the skin surrounding the eyes.

Conversely, intense cognitive concentration often activates the corrugator supercilii, gently knitting the eyebrows together in focused contemplation.

By combining neural networks for emotion classification with real-time gaze tracking, modern computer vision can observe both what you read and how you respond to every idea.`,
  },
  {
    title: "Daily Focus & Quick Notes",
    text: `• Note 1: Review project architecture and real-time gaze tracking pipeline.
• Note 2: Verify camera preview placement at top right and features displayed below.
• Note 3: Test copy, paste, and insert words functionality in the notepad editor.
• Note 4: Confirm calibrated confidence and facial expression analysis work simultaneously.
• Note 5: Observe how gaze focus smoothly follows across each reading line as you read down the page.`,
  },
];

export class NotepadController {
  private containerEl: HTMLElement;
  private editorEl: HTMLElement;
  private reticleEl: HTMLElement;
  private toastEl: HTMLElement;
  private tracker: GazeTracker;

  private currentFontSizeIndex = 1; // default Medium (20px)
  private fontSizes = ["text-base", "text-lg", "text-xl", "text-2xl"];
  private highlightEnabled = true;
  private reticleEnabled = true;
  private activeLineIndex = -1;
  private sampleIndex = 0;

  constructor(tracker: GazeTracker) {
    this.tracker = tracker;
    this.containerEl = document.getElementById("notepad-container")!;
    this.editorEl = document.getElementById("notepad-editor")!;
    this.reticleEl = document.getElementById("gaze-reticle")!;
    this.toastEl = document.getElementById("notepad-toast")!;

    this.initEditor();
    this.setupListeners();
  }

  private initEditor(): void {
    // Load initial sample text
    this.loadPassage(0);
  }

  public getContainer(): HTMLElement {
    return this.containerEl;
  }

  private setupListeners(): void {
    // Copy button
    const copyBtn = document.getElementById("notepad-copy-btn");
    copyBtn?.addEventListener("click", () => this.copyToClipboard());

    // Paste button
    const pasteBtn = document.getElementById("notepad-paste-btn");
    pasteBtn?.addEventListener("click", () => this.pasteFromClipboard());

    // Insert word button
    const insertBtn = document.getElementById("notepad-insert-btn");
    insertBtn?.addEventListener("click", () => this.promptInsertWord());

    // Sample text button
    const sampleBtn = document.getElementById("notepad-sample-btn");
    sampleBtn?.addEventListener("click", () => {
      this.sampleIndex = (this.sampleIndex + 1) % SAMPLE_PASSAGES.length;
      this.loadPassage(this.sampleIndex);
    });

    // Clear button
    const clearBtn = document.getElementById("notepad-clear-btn");
    clearBtn?.addEventListener("click", () => {
      if (confirm("Are you sure you want to clear the notepad?")) {
        this.setText("");
        this.showToast("Notepad cleared");
      }
    });

    // Font size buttons
    const fontDecBtn = document.getElementById("notepad-font-dec");
    fontDecBtn?.addEventListener("click", () => this.adjustFontSize(-1));

    const fontIncBtn = document.getElementById("notepad-font-inc");
    fontIncBtn?.addEventListener("click", () => this.adjustFontSize(1));

    // Highlight toggle
    const toggleHighlightBtn = document.getElementById("notepad-toggle-highlight");
    toggleHighlightBtn?.addEventListener("click", () => {
      this.highlightEnabled = !this.highlightEnabled;
      toggleHighlightBtn.classList.toggle("bg-indigo-600/30", this.highlightEnabled);
      toggleHighlightBtn.classList.toggle("text-indigo-300", this.highlightEnabled);
      this.showToast(this.highlightEnabled ? "Gaze highlight: ON" : "Gaze highlight: OFF");
      if (!this.highlightEnabled) {
        this.clearLineHighlights();
      }
    });

    // Reticle toggle
    const toggleReticleBtn = document.getElementById("notepad-toggle-reticle");
    toggleReticleBtn?.addEventListener("click", () => {
      this.reticleEnabled = !this.reticleEnabled;
      toggleReticleBtn.classList.toggle("bg-indigo-600/30", this.reticleEnabled);
      toggleReticleBtn.classList.toggle("text-indigo-300", this.reticleEnabled);
      this.reticleEl.style.display = this.reticleEnabled ? "block" : "none";
      this.showToast(this.reticleEnabled ? "Gaze reticle: ON" : "Gaze reticle: OFF");
    });

    // Center gaze calibration button
    const centerGazeBtn = document.getElementById("notepad-center-gaze");
    centerGazeBtn?.addEventListener("click", () => this.startCenterCalibration());

    // Sensitivity slider
    const sensSlider = document.getElementById("gaze-sens-slider") as HTMLInputElement | null;
    const sensVal = document.getElementById("gaze-sens-value");
    sensSlider?.addEventListener("input", () => {
      const val = parseFloat(sensSlider.value);
      this.tracker.setSensitivity(val);
      if (sensVal) sensVal.textContent = `${val.toFixed(1)}x`;
    });

    // Editor input listener to update line numbers and stats
    this.editorEl.addEventListener("input", () => {
      this.refreshLines();
      this.updateStats();
    });

    // Editor keyboard shortcuts
    this.editorEl.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        document.execCommand("insertText", false, "    ");
      }
    });
  }

  private adjustFontSize(delta: number): void {
    const nextIdx = Math.max(0, Math.min(this.fontSizes.length - 1, this.currentFontSizeIndex + delta));
    if (nextIdx !== this.currentFontSizeIndex) {
      this.fontSizes.forEach((cls) => this.editorEl.classList.remove(cls));
      this.currentFontSizeIndex = nextIdx;
      this.editorEl.classList.add(this.fontSizes[this.currentFontSizeIndex]!);
      this.showToast(`Font size: ${["Small", "Medium", "Large", "X-Large"][nextIdx]}`);
    }
  }

  public loadPassage(idx: number): void {
    const p = SAMPLE_PASSAGES[idx];
    if (!p) return;
    this.setText(p.text);
    this.showToast(`Loaded: "${p.title}"`);
  }

  public getText(): string {
    const lines = Array.from(this.editorEl.querySelectorAll("[data-notepad-line]"))
      .map((el) => (el as HTMLElement).innerText);
    return lines.length > 0 ? lines.join("\n") : this.editorEl.innerText;
  }

  public setText(text: string): void {
    const paragraphs = text.split("\n");
    this.editorEl.innerHTML = "";
    paragraphs.forEach((p, idx) => {
      const lineDiv = document.createElement("div");
      lineDiv.setAttribute("data-notepad-line", String(idx));
      lineDiv.className =
        "line-row flex items-start gap-3 py-1.5 px-3 rounded-lg transition-all duration-150 border border-transparent";
      
      const numSpan = document.createElement("span");
      numSpan.className =
        "select-none font-mono text-xs text-neutral-500 font-semibold pt-1 min-w-[24px] text-right";
      numSpan.textContent = String(idx + 1).padStart(2, "0");

      const textSpan = document.createElement("span");
      textSpan.className = "flex-1 outline-none text-neutral-100 leading-relaxed";
      textSpan.textContent = p || "\u00A0"; // Non-breaking space for empty lines

      lineDiv.appendChild(numSpan);
      lineDiv.appendChild(textSpan);
      this.editorEl.appendChild(lineDiv);
    });

    this.activeLineIndex = -1;
    this.updateStats();
  }

  private refreshLines(): void {
    // Ensure all children have line numbers and data attributes
    const rows = this.editorEl.querySelectorAll(".line-row");
    if (rows.length === 0) {
      // Re-parse flat text if structure was broken by paste
      const text = this.editorEl.innerText;
      this.setText(text);
    }
  }

  public insertWordAtCurrent(word: string): void {
    if (!word) return;
    // Insert into current active line or append to editor
    const activeEl = this.editorEl.querySelector(`[data-notepad-line="${this.activeLineIndex}"] .flex-1`);
    if (activeEl) {
      activeEl.textContent = (activeEl.textContent ?? "") + " " + word;
    } else {
      // Append a new line or add to last line
      const lines = this.editorEl.querySelectorAll(".line-row .flex-1");
      if (lines.length > 0) {
        const last = lines[lines.length - 1]!;
        last.textContent = (last.textContent ?? "") + " " + word;
      } else {
        this.setText(word);
      }
    }
    this.updateStats();
    this.showToast(`Inserted "${word}"`);
  }

  public promptInsertWord(): void {
    const word = window.prompt("Enter word or phrase to insert into notepad:", "Focus");
    if (word !== null && word.trim().length > 0) {
      this.insertWordAtCurrent(word.trim());
    }
  }

  public async copyToClipboard(): Promise<void> {
    const text = this.getText();
    if (!text.trim()) {
      this.showToast("Notepad is empty");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      this.showToast("✓ Copied all text to clipboard!");
    } catch {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      this.showToast("✓ Copied to clipboard!");
    }
  }

  public async pasteFromClipboard(): Promise<void> {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        // Append or replace
        const current = this.getText();
        const combined = current.trim() ? `${current}\n\n${text}` : text;
        this.setText(combined);
        this.showToast("✓ Pasted from clipboard!");
      } else {
        this.showToast("Clipboard is empty");
      }
    } catch {
      // If browser clipboard permission blocked, prompt user
      const pasted = window.prompt("Paste your text here (Ctrl+V):");
      if (pasted) {
        const current = this.getText();
        const combined = current.trim() ? `${current}\n\n${pasted}` : pasted;
        this.setText(combined);
        this.showToast("✓ Pasted successfully!");
      }
    }
  }

  public startCenterCalibration(): void {
    const centerBtn = document.getElementById("notepad-center-gaze");
    if (centerBtn) {
      centerBtn.classList.add("animate-pulse", "bg-rose-600", "text-white");
      centerBtn.textContent = "👀 Look at Center of Notepad...";
    }
    this.showToast("Look directly at the center of the notepad for 2 seconds...");

    setTimeout(() => {
      // Trigger calibration at zero
      this.tracker.resetCalibration();
      if (centerBtn) {
        centerBtn.classList.remove("animate-pulse", "bg-rose-600", "text-white");
        centerBtn.textContent = "🎯 Center Gaze (Calibrated!)";
        setTimeout(() => {
          centerBtn.textContent = "🎯 Center Gaze";
        }, 2500);
      }
      this.showToast("✓ Gaze center calibrated!");
    }, 1800);
  }

  private clearLineHighlights(): void {
    const lines = this.editorEl.querySelectorAll("[data-notepad-line]");
    lines.forEach((l) => {
      l.classList.remove(
        "bg-indigo-600/20",
        "border-indigo-500/80",
        "ring-1",
        "ring-indigo-500/40",
        "shadow-lg",
        "shadow-indigo-500/10",
      );
    });
  }

  /**
   * Update gaze tracking indicators based on real-time gaze result
   */
  public updateGaze(result: GazeResult): void {
    // 1. Update floating reticle
    if (this.reticleEnabled) {
      this.reticleEl.style.display = "block";
      this.reticleEl.style.transform = `translate3d(${result.screenX}px, ${result.screenY}px, 0)`;

      if (result.isOnNotepad) {
        this.reticleEl.classList.remove("border-neutral-500", "bg-neutral-500/20");
        this.reticleEl.classList.add("border-indigo-400", "bg-indigo-500/30", "scale-110");
      } else {
        this.reticleEl.classList.remove("border-indigo-400", "bg-indigo-500/30", "scale-110");
        this.reticleEl.classList.add("border-neutral-500", "bg-neutral-500/20");
      }
    }

    // 2. Line highlighting
    if (this.highlightEnabled && result.isOnNotepad && result.targetLineIndex >= 0) {
      if (result.targetLineIndex !== this.activeLineIndex) {
        this.clearLineHighlights();
        this.activeLineIndex = result.targetLineIndex;
        const targetEl = this.editorEl.querySelector(`[data-notepad-line="${this.activeLineIndex}"]`);
        if (targetEl) {
          targetEl.classList.add(
            "bg-indigo-600/20",
            "border-indigo-500/80",
            "ring-1",
            "ring-indigo-500/40",
            "shadow-lg",
            "shadow-indigo-500/10",
          );
        }
      }
    } else if (!result.isOnNotepad && this.activeLineIndex !== -1) {
      this.clearLineHighlights();
      this.activeLineIndex = -1;
    }

    // 3. Update stats header pill
    this.updateGazeHUD(result);
  }

  private updateGazeHUD(result: GazeResult): void {
    const gazeStatusPill = document.getElementById("gaze-hud-status");
    const gazeTargetPill = document.getElementById("gaze-hud-target");
    const gazeDwellPill = document.getElementById("gaze-hud-dwell");

    if (gazeStatusPill) {
      if (result.isOnNotepad) {
        gazeStatusPill.textContent = "👁️ Tracking: Reading Notepad";
        gazeStatusPill.className =
          "px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
      } else if (result.lookingRegion === "camera") {
        gazeStatusPill.textContent = "📷 Looking at Camera";
        gazeStatusPill.className =
          "px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40";
      } else if (result.lookingRegion === "features") {
        gazeStatusPill.textContent = "📊 Looking at Analytics";
        gazeStatusPill.className =
          "px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40";
      } else {
        gazeStatusPill.textContent = "👀 Eye Tracking Active";
        gazeStatusPill.className =
          "px-2.5 py-1 rounded-full text-xs font-semibold bg-neutral-800 text-neutral-300 border border-neutral-700";
      }
    }

    if (gazeTargetPill) {
      if (result.isOnNotepad && result.targetLineIndex >= 0) {
        gazeTargetPill.textContent = `Line ${result.targetLineIndex + 1} of ${result.totalLines}`;
      } else {
        gazeTargetPill.textContent = result.statusText;
      }
    }

    if (gazeDwellPill) {
      const sec = (result.dwellTimeMs / 1000).toFixed(1);
      gazeDwellPill.textContent = `Focus: ${sec}s`;
    }
  }

  private updateStats(): void {
    const text = this.getText();
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    const lines = this.editorEl.querySelectorAll("[data-notepad-line]").length;

    const wordCountEl = document.getElementById("notepad-word-count");
    const charCountEl = document.getElementById("notepad-char-count");
    const lineCountEl = document.getElementById("notepad-line-count");

    if (wordCountEl) wordCountEl.textContent = `${words} words`;
    if (charCountEl) charCountEl.textContent = `${chars} chars`;
    if (lineCountEl) lineCountEl.textContent = `${lines} lines`;
  }

  public showToast(msg: string): void {
    if (!this.toastEl) return;
    this.toastEl.textContent = msg;
    this.toastEl.classList.remove("opacity-0", "translate-y-2");
    this.toastEl.classList.add("opacity-100", "translate-y-0");

    setTimeout(() => {
      this.toastEl.classList.remove("opacity-100", "translate-y-0");
      this.toastEl.classList.add("opacity-0", "translate-y-2");
    }, 2200);
  }
}

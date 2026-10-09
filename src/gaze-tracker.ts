// Real-time Eye & Gaze Tracking engine using MediaPipe FaceLandmarker
// Uses iris center landmarks (468, 473), eye corners, eyelids, head orientation,
// and ARKit blendshapes with EMA smoothing and screen/notepad projection.

import type { FaceFeatures } from "./face-pipeline";

export interface GazePoint {
  x: number; // Viewport X in pixels
  y: number; // Viewport Y in pixels
}

export interface GazeResult {
  // Screen/viewport pixel coordinates
  screenX: number;
  screenY: number;
  // Normalized coordinates: -1 to 1 (left to right, top to bottom)
  normX: number;
  normY: number;
  // Notepad-relative coordinates (if over notepad)
  isOnNotepad: boolean;
  notepadRelX: number; // 0 to 1
  notepadRelY: number; // 0 to 1
  targetLineIndex: number; // 0-indexed line being read
  totalLines: number;
  // Detected region
  lookingRegion: "notepad" | "camera" | "features" | "top" | "away";
  statusText: string;
  // Raw eye signals
  leftEyeOpen: number; // 0 (closed) to 1 (wide)
  rightEyeOpen: number;
  isBlinking: boolean;
  horizontalDirection: "left" | "center" | "right";
  verticalDirection: "up" | "center" | "down";
  // Attention metrics
  attentionScore: number; // 0 to 100
  dwellTimeMs: number; // Time focused on current area
  confidence: number;
}

export class GazeTracker {
  // Calibration offsets and scale
  private offsetX = 0;
  private offsetY = 0;
  private sensitivity = 1.35; // default sensitivity multiplier

  // Exponential moving average for silky smooth gaze cursor
  private smoothedNormX = 0;
  private smoothedNormY = 0;
  private smoothedScreenX = 0;
  private smoothedScreenY = 0;
  private alpha = 0.22; // Smoothing factor (lower = smoother, higher = faster)

  // Tracking state
  private lastTargetLine = -1;
  private lineFocusStartTime = performance.now();
  private lastUpdateTime = performance.now();
  private isCalibrated = false;

  public isGazeCalibrated(): boolean {
    return this.isCalibrated;
  }

  public getLastUpdateTime(): number {
    return this.lastUpdateTime;
  }

  constructor() {
    this.smoothedScreenX = window.innerWidth / 2;
    this.smoothedScreenY = window.innerHeight / 2;
  }

  public setSensitivity(multiplier: number): void {
    this.sensitivity = Math.max(0.5, Math.min(3.0, multiplier));
  }

  public getSensitivity(): number {
    return this.sensitivity;
  }

  // Recalibrate center: sets current raw gaze direction as (0, 0)
  public calibrateCenter(currentRawX: number, currentRawY: number): void {
    this.offsetX = currentRawX;
    this.offsetY = currentRawY;
    this.isCalibrated = true;
  }

  // Reset calibration
  public resetCalibration(): void {
    this.offsetX = 0;
    this.offsetY = 0;
    this.isCalibrated = false;
  }

  /**
   * Process face landmarks and blendshapes to compute gaze vector and target in the notepad.
   */
  public process(
    face: FaceFeatures,
    notepadEl: HTMLElement | null,
    cameraEl: HTMLElement | null,
    nowMs: number = performance.now(),
  ): GazeResult {
    const { landmarks, blendshapes } = face;
    const lms = landmarks;
    const bs = blendshapes;

    // 1. Iris and Eye Corner Landmarks
    // Left iris center = 468, right iris center = 473
    // Note: Landmarks are normalized (0 to 1) in video coordinates.
    // Video is mirrored for display, so user's physical left eye is landmark 468.
    const hasIris = lms.length >= 478 * 3;

    // Eye blendshapes (0 to 1)
    const lookInL = bs["eyeLookInLeft"] ?? 0;
    const lookOutL = bs["eyeLookOutLeft"] ?? 0;
    const lookUpL = bs["eyeLookUpLeft"] ?? 0;
    const lookDownL = bs["eyeLookDownLeft"] ?? 0;

    const lookInR = bs["eyeLookInRight"] ?? 0;
    const lookOutR = bs["eyeLookOutRight"] ?? 0;
    const lookUpR = bs["eyeLookUpRight"] ?? 0;
    const lookDownR = bs["eyeLookDownRight"] ?? 0;

    const blinkL = bs["eyeBlinkLeft"] ?? 0;
    const blinkR = bs["eyeBlinkRight"] ?? 0;
    const isBlinking = (blinkL + blinkR) / 2 > 0.45;

    // Horizontal eye gaze from blendshapes:
    // When looking screen-left: user's physical right eye looks OUT, left eye looks IN.
    // When looking screen-right: user's physical left eye looks OUT, right eye looks IN.
    const bsHorizontal = (lookOutR + lookInL) - (lookOutL + lookInR);
    const bsVertical = ((lookDownL + lookDownR) / 2) - ((lookUpL + lookUpR) / 2);

    let landmarkHoriz = 0;
    let landmarkVert = 0;

    if (hasIris) {
      // Landmark coords
      // Subject right eye (viewer's left before mirror): outer corner 33, inner corner 133, iris 468
      const x33 = lms[33 * 3];
      const x133 = lms[133 * 3];
      const x468 = lms[468 * 3];
      const y159 = lms[159 * 3 + 1]; // upper lid
      const y145 = lms[145 * 3 + 1]; // lower lid
      const y468 = lms[468 * 3 + 1];

      // Subject left eye (viewer's right before mirror): inner corner 362, outer corner 263, iris 473
      const x362 = lms[362 * 3];
      const x263 = lms[263 * 3];
      const x473 = lms[473 * 3];
      const y386 = lms[386 * 3 + 1]; // upper lid
      const y374 = lms[374 * 3 + 1]; // lower lid
      const y473 = lms[473 * 3 + 1];

      const rEyeSpan = Math.abs(x133 - x33) || 0.01;
      const lEyeSpan = Math.abs(x263 - x362) || 0.01;

      // Relative iris position (0 = inner, 1 = outer or vice versa)
      const rEyeIrisRelX = (x468 - Math.min(x33, x133)) / rEyeSpan - 0.5;
      const lEyeIrisRelX = (x473 - Math.min(x362, x263)) / lEyeSpan - 0.5;

      landmarkHoriz = -(rEyeIrisRelX + lEyeIrisRelX); // flip for mirror

      // Vertical iris relative to lids
      const rEyeLidSpan = Math.abs(y145 - y159) || 0.01;
      const lEyeLidSpan = Math.abs(y374 - y386) || 0.01;
      const rEyeIrisRelY = (y468 - Math.min(y159, y145)) / rEyeLidSpan - 0.5;
      const lEyeIrisRelY = (y473 - Math.min(y386, y374)) / lEyeLidSpan - 0.5;
      landmarkVert = (rEyeIrisRelY + lEyeIrisRelY) / 2;
    }

    // Head orientation (nose relative to face center)
    // Nose tip: 1, Chin: 152, Forehead: 10, Left temple: 234, Right temple: 454
    const noseX = lms[1 * 3];
    const noseY = lms[1 * 3 + 1];
    const templeL = lms[234 * 3];
    const templeR = lms[454 * 3];
    const foreheadY = lms[10 * 3 + 1];
    const chinY = lms[152 * 3 + 1];

    const faceCenterNormX = (templeL + templeR) / 2;
    const headYaw = -(noseX - faceCenterNormX) * 2.2; // mirrored

    const faceCenterNormY = (foreheadY + chinY) / 2;
    const headPitch = (noseY - faceCenterNormY) * 2.0;

    // Combine eyes and head pose for total gaze vector
    const rawGazeX = (bsHorizontal * 1.5 + landmarkHoriz * 0.8 + headYaw * 0.7);
    const rawGazeY = (bsVertical * 1.8 + landmarkVert * 0.8 + headPitch * 0.6);

    // Apply calibration offset
    const centeredX = (rawGazeX - this.offsetX) * this.sensitivity;
    const centeredY = (rawGazeY - this.offsetY) * this.sensitivity;

    // Smooth normalized gaze with EMA
    this.smoothedNormX = this.smoothedNormX * (1 - this.alpha) + centeredX * this.alpha;
    this.smoothedNormY = this.smoothedNormY * (1 - this.alpha) + centeredY * this.alpha;

    // Viewport dimensions
    const vWidth = window.innerWidth;
    const vHeight = window.innerHeight;

    // Project normalized gaze [-1, 1] to screen coordinates
    // Center of screen is (vWidth / 2, vHeight / 2)
    // Looking left -> smaller screenX; looking right -> larger screenX.
    const targetScreenX = Math.max(0, Math.min(vWidth, vWidth / 2 + this.smoothedNormX * (vWidth * 0.65)));
    const targetScreenY = Math.max(0, Math.min(vHeight, vHeight / 2 + this.smoothedNormY * (vHeight * 0.65)));

    this.smoothedScreenX = this.smoothedScreenX * (1 - this.alpha) + targetScreenX * this.alpha;
    this.smoothedScreenY = this.smoothedScreenY * (1 - this.alpha) + targetScreenY * this.alpha;

    // Detect target region & Notepad reading line
    let isOnNotepad = false;
    let notepadRelX = 0;
    let notepadRelY = 0;
    let targetLineIndex = -1;
    let totalLines = 1;
    let lookingRegion: "notepad" | "camera" | "features" | "top" | "away" = "away";

    if (notepadEl) {
      const rect = notepadEl.getBoundingClientRect();
      const padding = 20; // soft bounding box margin
      if (
        this.smoothedScreenX >= rect.left - padding &&
        this.smoothedScreenX <= rect.right + padding &&
        this.smoothedScreenY >= rect.top - padding &&
        this.smoothedScreenY <= rect.bottom + padding
      ) {
        isOnNotepad = true;
        lookingRegion = "notepad";

        // Clamp to notepad inner box
        const innerX = Math.max(0, Math.min(rect.width, this.smoothedScreenX - rect.left));
        const innerY = Math.max(0, Math.min(rect.height, this.smoothedScreenY - rect.top));
        notepadRelX = innerX / rect.width;
        notepadRelY = innerY / rect.height;

        // Calculate line index if lines are rendered
        const lineElements = notepadEl.querySelectorAll<HTMLElement>("[data-notepad-line]");
        totalLines = lineElements.length;

        if (totalLines > 0) {
          // Find which line element is closest to smoothedScreenY
          let minDistance = Infinity;
          let bestIdx = 0;

          lineElements.forEach((line, idx) => {
            const lineRect = line.getBoundingClientRect();
            const lineCenterY = (lineRect.top + lineRect.bottom) / 2;
            const dist = Math.abs(this.smoothedScreenY - lineCenterY);
            if (dist < minDistance) {
              minDistance = dist;
              bestIdx = idx;
            }
          });
          targetLineIndex = bestIdx;
        } else {
          // Approximate by relative Y
          targetLineIndex = Math.floor(notepadRelY * 10);
        }
      }
    }

    if (!isOnNotepad && cameraEl) {
      const cRect = cameraEl.getBoundingClientRect();
      if (
        this.smoothedScreenX >= cRect.left - 30 &&
        this.smoothedScreenX <= cRect.right + 30 &&
        this.smoothedScreenY >= cRect.top - 30 &&
        this.smoothedScreenY <= cRect.bottom + 30
      ) {
        lookingRegion = "camera";
      } else if (this.smoothedScreenX > vWidth * 0.55 && this.smoothedScreenY > cRect.bottom) {
        lookingRegion = "features";
      } else if (this.smoothedScreenY < 70) {
        lookingRegion = "top";
      }
    }

    // Dwell time on current line / region
    if (targetLineIndex !== this.lastTargetLine) {
      this.lastTargetLine = targetLineIndex;
      this.lineFocusStartTime = nowMs;
    }
    const dwellTimeMs = nowMs - this.lineFocusStartTime;

    // Gaze directions
    const hThreshold = 0.15;
    const vThreshold = 0.15;
    const horizontalDirection: "left" | "center" | "right" =
      this.smoothedNormX < -hThreshold ? "left" : this.smoothedNormX > hThreshold ? "right" : "center";
    const verticalDirection: "up" | "center" | "down" =
      this.smoothedNormY < -vThreshold ? "up" : this.smoothedNormY > vThreshold ? "down" : "center";

    // Eye openness (1 - blink)
    const leftEyeOpen = Math.max(0, 1 - blinkL);
    const rightEyeOpen = Math.max(0, 1 - blinkR);

    // Attention score based on stability and dwell
    let attentionScore = 70;
    if (isOnNotepad) {
      attentionScore = Math.min(100, 75 + Math.min(25, Math.floor(dwellTimeMs / 200)));
    } else if (lookingRegion === "camera" || lookingRegion === "features") {
      attentionScore = 80;
    } else {
      attentionScore = Math.max(20, 50 - Math.floor(dwellTimeMs / 1000));
    }

    // Status description string
    let statusText = "Scanning screen";
    if (isBlinking) {
      statusText = "Blinking";
    } else if (isOnNotepad) {
      statusText = `Reading line ${targetLineIndex + 1} of ${Math.max(1, totalLines)}`;
    } else if (lookingRegion === "camera") {
      statusText = "Looking at Camera";
    } else if (lookingRegion === "features") {
      statusText = "Looking at Analytics below camera";
    } else if (horizontalDirection === "left") {
      statusText = "Looking left of notepad";
    } else if (horizontalDirection === "right") {
      statusText = "Looking right side";
    }

    this.lastUpdateTime = nowMs;

    return {
      screenX: this.smoothedScreenX,
      screenY: this.smoothedScreenY,
      normX: this.smoothedNormX,
      normY: this.smoothedNormY,
      isOnNotepad,
      notepadRelX,
      notepadRelY,
      targetLineIndex,
      totalLines,
      lookingRegion,
      statusText,
      leftEyeOpen,
      rightEyeOpen,
      isBlinking,
      horizontalDirection,
      verticalDirection,
      attentionScore,
      dwellTimeMs,
      confidence: hasIris ? 0.92 : 0.7,
    };
  }
}

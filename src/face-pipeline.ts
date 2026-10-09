// MediaPipe FaceLandmarker wrapper. The instance is expensive to init
// (model + wasm download, GPU context warmup) -- callers should cache it.
//
// We use VIDEO running mode rather than LIVE_STREAM. Both work for a webcam,
// but VIDEO returns results synchronously, which fits our rAF-driven loop in
// main.ts and keeps the rest of the pipeline as pure functions over feature
// data. Swap to LIVE_STREAM only if profiling shows the
// detect call is stalling frames.

import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const BBOX_PADDING = 0.08; // 8% padding around landmark hull

export interface FaceFeatures {
  bbox: { x: number; y: number; w: number; h: number }; // pixel coords
  landmarks: Float32Array; // length 478 * 3, [x0,y0,z0, x1,y1,z1, ...] normalized
  blendshapes: Record<string, number>; // 52 ARKit categories
}

export interface FacePipeline {
  process(video: HTMLVideoElement, timestampMs: number): FaceFeatures[];
  close(): void;
}

export async function createFacePipeline(): Promise<FacePipeline> {
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);
  const modelAssetPath =
    import.meta.env.VITE_MEDIAPIPE_MODEL_URL ?? DEFAULT_MODEL_URL;

  // Pick the delegate based on the platform. On mobile Android Chrome the
  // WebGL/GPU delegate frequently *succeeds* at construction but then returns
  // zero detections silently -- a known MediaPipe issue tracked across
  // multiple github threads. CPU delegate is reliable on mobile WASM.
  // Laptops keep the GPU delegate as before; desktop path is unchanged.
  const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  const preferredDelegate: "GPU" | "CPU" = isMobile ? "CPU" : "GPU";
  const buildLandmarker = (delegate: "GPU" | "CPU"): Promise<FaceLandmarker> =>
    FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: "VIDEO",
      numFaces: 5,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    });
  let landmarker: FaceLandmarker;
  try {
    landmarker = await buildLandmarker(preferredDelegate);
  } catch (err) {
    const fallback = preferredDelegate === "GPU" ? "CPU" : "GPU";
    console.warn(
      `FaceLandmarker ${preferredDelegate} delegate failed, falling back to ${fallback}:`,
      err,
    );
    landmarker = await buildLandmarker(fallback);
  }

  return {
    process(video, timestampMs) {
      if (video.readyState < 2 || video.videoWidth === 0) return [];
      const result: FaceLandmarkerResult = landmarker.detectForVideo(
        video,
        timestampMs,
      );
      return toFeatures(result, video.videoWidth, video.videoHeight);
    },
    close() {
      landmarker.close();
    },
  };
}

function toFeatures(
  result: FaceLandmarkerResult,
  videoW: number,
  videoH: number,
): FaceFeatures[] {
  const out: FaceFeatures[] = [];
  const landmarksList = result.faceLandmarks ?? [];
  const blendshapesList = result.faceBlendshapes ?? [];

  for (let i = 0; i < landmarksList.length; i++) {
    const lms = landmarksList[i];
    if (!lms) continue;

    const flat = new Float32Array(lms.length * 3);
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (let j = 0; j < lms.length; j++) {
      const p = lms[j]!;
      flat[j * 3 + 0] = p.x;
      flat[j * 3 + 1] = p.y;
      flat[j * 3 + 2] = p.z;
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    const padX = (maxX - minX) * BBOX_PADDING;
    const padY = (maxY - minY) * BBOX_PADDING;
    const x = Math.max(0, (minX - padX) * videoW);
    const y = Math.max(0, (minY - padY) * videoH);
    const w = Math.min(videoW - x, (maxX - minX + 2 * padX) * videoW);
    const h = Math.min(videoH - y, (maxY - minY + 2 * padY) * videoH);

    const blendshapes: Record<string, number> = {};
    const bs = blendshapesList[i];
    if (bs) {
      for (const cat of bs.categories) {
        blendshapes[cat.categoryName] = cat.score;
      }
    }

    out.push({ bbox: { x, y, w, h }, landmarks: flat, blendshapes });
  }
  return out;
}

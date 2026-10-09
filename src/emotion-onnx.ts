// HSEmotion enet_b0_8_va_mtl ONNX classifier.
// Source: github.com/HSE-asavchenko/face-emotion-recognition (Apache-2.0).
// Trained on AffectNet 8-cls; outputs 8 emotion logits + valence + arousal in
// a single [1, 10] tensor. EfficientNet-B0 backbone, 224x224 RGB input,
// ImageNet mean/std normalization, NCHW layout.
//
// Class order from HSEmotion (alphabetical, AffectNet convention):
//   0:Anger 1:Contempt 2:Disgust 3:Fear 4:Happiness 5:Neutral 6:Sadness 7:Surprise

import * as ort from "onnxruntime-web";
import { type Emotion, type EmotionReadout } from "./emotion-head";

const INPUT_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406] as const;
const STD = [0.229, 0.224, 0.225] as const;

const HSE_INDEX_TO_EMOTION: readonly Emotion[] = [
  "angry",
  "contempt",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprised",
];

const DEFAULT_MODEL_URL =
  "https://cdn.jsdelivr.net/gh/HSE-asavchenko/face-emotion-recognition@main/models/affectnet_emotions/onnx/enet_b0_8_va_mtl.onnx";

// Multi-threaded WASM needs SharedArrayBuffer (COOP/COEP headers); HF Spaces
// static deploys typically don't set those, so single-threaded WASM is the
// safe default. SIMD is fine without cross-origin isolation.
const WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EmotionClassifier {
  classify(video: HTMLVideoElement, bbox: BBox): Promise<EmotionReadout>;
  close(): void;
}

export async function createEmotionClassifier(): Promise<EmotionClassifier> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = WASM_BASE;

  const url = import.meta.env.VITE_HSEMOTION_MODEL_URL ?? DEFAULT_MODEL_URL;
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const inputName = session.inputNames[0];
  if (!inputName) throw new Error("HSEmotion ONNX has no input");

  // Plain HTMLCanvasElement (off-DOM) instead of OffscreenCanvas: works on
  // every browser including iOS Safari pre-16.4. Same perf for our use case
  // since we never transfer the canvas to a worker.
  const crop = document.createElement("canvas");
  crop.width = INPUT_SIZE;
  crop.height = INPUT_SIZE;
  const cropCtx = crop.getContext("2d", { willReadFrequently: true });
  if (!cropCtx) throw new Error("canvas 2d context unavailable");

  return {
    async classify(video, bbox) {
      // Square crop around bbox center, clamped to video bounds.
      const side = Math.max(bbox.w, bbox.h);
      const cx = bbox.x + bbox.w / 2;
      const cy = bbox.y + bbox.h / 2;
      const sx = Math.max(0, Math.min(video.videoWidth - side, cx - side / 2));
      const sy = Math.max(0, Math.min(video.videoHeight - side, cy - side / 2));
      const sw = Math.min(side, video.videoWidth - sx);
      const sh = Math.min(side, video.videoHeight - sy);

      cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);
      const img = cropCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
      const N = INPUT_SIZE * INPUT_SIZE;
      const data = new Float32Array(3 * N);
      const px = img.data;
      for (let i = 0; i < N; i++) {
        const j = i * 4;
        data[i] = (px[j]! / 255 - MEAN[0]) / STD[0];
        data[N + i] = (px[j + 1]! / 255 - MEAN[1]) / STD[1];
        data[2 * N + i] = (px[j + 2]! / 255 - MEAN[2]) / STD[2];
      }

      const input = new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const result = await session.run({ [inputName]: input });

      // The MTL model concatenates [emotion(8), valence, arousal] into a single
      // length-10 vector. Defensive: also handle a future split-output variant.
      const logits: Record<Emotion, number> = {
        neutral: 0, happy: 0, sad: 0, surprised: 0,
        angry: 0, disgust: 0, fear: 0, contempt: 0,
      };
      let valence = 0;
      let arousal = 0;

      const firstName = session.outputNames[0]!;
      const first = result[firstName]!.data as Float32Array;
      for (let i = 0; i < 8; i++) {
        logits[HSE_INDEX_TO_EMOTION[i]!] = first[i]!;
      }
      if (first.length >= 10) {
        valence = first[8]!;
        arousal = first[9]!;
      } else if (session.outputNames.length >= 3) {
        valence = (result[session.outputNames[1]!]!.data as Float32Array)[0]!;
        arousal = (result[session.outputNames[2]!]!.data as Float32Array)[0]!;
      }

      return { logits, valence, arousal };
    },
    close() {
      void session.release();
    },
  };
}

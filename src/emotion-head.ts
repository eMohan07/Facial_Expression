// Emotion type definitions. The actual classifier lives in emotion-onnx.ts
// (HSEmotion enet_b0_8_va_mtl). This file used to host a hand-tuned blendshape
// head; it was retired once the HSEmotion ONNX classifier was adopted.

export const EMOTIONS = [
  "neutral",
  "happy",
  "sad",
  "surprised",
  "angry",
  "disgust",
  "fear",
  "contempt",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export interface EmotionReadout {
  logits: Record<Emotion, number>;
  valence: number; // -1..1
  arousal: number; // -1..1
}

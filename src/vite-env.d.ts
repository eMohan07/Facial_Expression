/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEMO_ANTHROPIC_API_KEY?: string;
  readonly VITE_EXPLAIN_PROXY_URL?: string;
  readonly VITE_ANTHROPIC_MODEL?: string;
  readonly VITE_MEDIAPIPE_MODEL_URL?: string;
  readonly VITE_HSEMOTION_MODEL_URL?: string;
  readonly VITE_PIPELINE_VARIANT?: "A" | "B";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

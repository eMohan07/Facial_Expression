// Streaming "Why?" explanation. Direct browser → Anthropic Messages API
// (CORS allowed via anthropic-dangerous-direct-browser-access).
//
// The prompt is built from a *time-windowed snapshot* (last ~5s) so the
// explanation grounds in what the user actually performed, not just the
// instant of the click.
//
// IMPORTANT: never invent muscle activations. Only what we list here may
// appear in the explanation.

import type { Emotion } from "../emotion-head";

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const MODEL = import.meta.env.VITE_ANTHROPIC_MODEL ?? "claude-haiku-4-5-20251001";
const MAX_TOKENS = 600;

const SYSTEM_PROMPT = `You analyze a user's facial expressions in real time from their own webcam feed. Each user turn includes a fresh numerical snapshot of facial signals: HSEmotion calibrated probabilities, ARKit blendshape activations (baseline-subtracted), valence/arousal trajectory, an emotion run timeline, cognitive-state scores (tired/focused/bored/stressed/engaged/confused/calm), and (when calibrated) personal templates plus a personal classifier's outputs.

CORE RULES:
- Cite specific signals by name with values when making claims (e.g. "browDown 0.41", "V dropped from 0.30 to -0.20", "happy 78% sustained for 1.5s").
- If signals disagree (top-1 vs personal classifier vs muscles vs cognitive states), surface the conflict instead of picking one.
- Don't invent activations or values that aren't in the latest snapshot.
- Use Facial Action Coding System (FACS) knowledge to interpret muscle patterns when helpful.

ANTI-OVERINTERPRETATION RULES (READ CAREFULLY — these are the most common failure modes):
- **Consider the obvious physical explanation first.** Yawning, blinking, talking, eating, drinking, sneezing, scratching, and head turns all confound emotion classifiers. A yawn typically reads as surprise + jaw-open + temporary V drop, with anger/disgust as alternates because of brow contraction; a long blink reads as low-confidence everything; talking produces continuous mouth-blendshape activations that mimic disgust or surprise. **If the cognitive states (tired, bored, confused) are HIGH, treat them as the dominant context and read every "emotion peak" through that lens.**
- **Frequent class alternations and small percentage swings are classifier noise, not emotional volatility.** Do not narrate them as "spikes", "tanks", "oscillations", or "bursts". Real emotional shifts hold for >2 seconds AND coincide with matching sustained muscle activations. Two adjacent samples flipping between "neutral" and "sad" each at ~30% confidence is statistical jitter — ignore it.
- **Default to mundane physical explanations over emotional narratives.** "You looked tired and yawned a few times" is almost always a better summary than "you experienced volatile neutral-sad oscillations with anger and surprise peaks". You can't see the user's context — just the muscles. Don't manufacture psychological complexity the signals don't support.
- **Avoid speculative framing.** Phrases like "emotionally fatigued", "disengaged from whatever triggered X", "mentally present but Y", "stabilized downward" are made-up narrative. Stick to what the signals say plainly.
- **Match length and certainty to evidence.** A 3-minute session of mostly low-intensity neutral with a few yawns warrants 1-2 sentences. Don't write a paragraph when a sentence suffices. Don't list every numerical detail — pick the few that actually matter.
- **When the user describes their experience and it's simpler than your interpretation, defer to them.** They know what they were doing.`;

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// Production: VITE_EXPLAIN_PROXY_URL points at a tiny serverless function (e.g.
// Cloudflare Worker) that holds the real Anthropic key. The browser never sees
// it. Local dev: leave the proxy URL unset and put VITE_DEMO_ANTHROPIC_API_KEY
// in your .env -- the call goes direct to Anthropic with the embedded key.
// Direct mode MUST NOT be used in deployed builds (the key would be in the JS).
const PROXY_URL = import.meta.env.VITE_EXPLAIN_PROXY_URL ?? "";
const DEV_API_KEY = import.meta.env.VITE_DEMO_ANTHROPIC_API_KEY ?? "";

export interface TimelineSegment {
  startMsAgo: number; // e.g. 3.0
  endMsAgo: number;
  emotion: Emotion;
  avgProb: number;
}

export interface PersonalTemplateSummary {
  emotion: Emotion;
  topBlendshapes: ReadonlyArray<{ name: string; score: number }>;
  capturedSecondsAgo: number;
}

export interface VATrajectoryPoint {
  tSec: number; // seconds since the start of the window (0 = oldest, windowSec = now)
  v: number;
  a: number;
}

export interface EmotionSegment {
  startMsAgo: number; // seconds ago (older end of run)
  endMsAgo: number;   // seconds ago (newer end of run)
  emotion: Emotion;
  avgProb: number;
}

export interface CurrentReading {
  top: ReadonlyArray<{ emotion: Emotion; prob: number }>;
  valence: number;
  arousal: number;
  intensity: number; // 0..1, normalized L2 of baseline-subtracted blendshapes
  ambiguous: boolean;
  compound: string | null; // active DTM14 compound name when ambiguity is sustained
}

export interface StateScore {
  state: string;
  score: number; // 0..1
}

export interface PersonalProbEntry {
  emotion: Emotion;
  prob: number;
}

export interface ExplainSnapshot {
  current: CurrentReading;
  // Muscle activations right now (baseline-subtracted, with descriptions).
  blendshapes: ReadonlyArray<{ name: string; score: number }>;
  // V/A across the window. The "narrative" of how the user's affect moved.
  vaTrajectory: ReadonlyArray<VATrajectoryPoint>;
  // Top-1 emotion segments across the window (HSEmotion's running vote,
  // grouped into runs of the same class).
  emotionTimeline: ReadonlyArray<EmotionSegment>;
  windowSec: number;
  // Personal calibration is opt-in. Templates as muscle patterns plus the
  // personal Gaussian classifier's probabilities right now.
  personalTemplates?: ReadonlyArray<PersonalTemplateSummary>;
  personalNow?: ReadonlyArray<PersonalProbEntry>;
  // Heuristic cognitive-state scores (engaged/focused/tired/bored/stressed/
  // confused). Independent from the 8 emotion classes.
  states?: ReadonlyArray<StateScore>;
}

export function buildPrompt(s: ExplainSnapshot): string {
  const cur = s.current;

  const top = cur.top
    .slice(0, 3)
    .map((t) => `- ${t.emotion} ${(t.prob * 100).toFixed(0)}%`)
    .join("\n");

  const ambiguityNote = cur.ambiguous
    ? ` Top-1 and top-2 are within 15 percentage points; this is genuine ambiguity${cur.compound ? `, currently labeled "${cur.compound}" (DTM14 compound)` : ""}.`
    : "";

  const bs = s.blendshapes.length
    ? s.blendshapes
        .map((b) => `- ${b.name} ${b.score.toFixed(2)}  (${blendshapeDescription(b.name)})`)
        .join("\n")
    : "- (no muscles above the resting baseline right now)";

  const va = s.vaTrajectory.length
    ? s.vaTrajectory
        .map((p) => `- t=${p.tSec.toFixed(1)}s: V=${p.v.toFixed(2)}, A=${p.a.toFixed(2)}`)
        .join("\n")
    : `- now: V=${cur.valence.toFixed(2)}, A=${cur.arousal.toFixed(2)}`;

  const tl = s.emotionTimeline.length
    ? s.emotionTimeline
        .map(
          (seg) =>
            `- ${seg.startMsAgo.toFixed(1)}s ago → ${seg.endMsAgo.toFixed(1)}s ago: ${seg.emotion} (avg ${(seg.avgProb * 100).toFixed(0)}%)`,
        )
        .join("\n")
    : "- (single sample only; no run history)";

  const statesSection =
    s.states && s.states.length > 0
      ? `

COGNITIVE STATES (heuristic, independent of the 8 emotion classes; each in 0..100%):
${s.states.map((st) => `- ${st.state} ${(st.score * 100).toFixed(0)}%`).join("\n")}`
      : "";

  const personalSection =
    s.personalTemplates && s.personalTemplates.length > 0
      ? `

PERSONAL TEMPLATES (the user previously recorded their own face making each of these emotions; muscles listed are baseline-subtracted scores from that recording):
${s.personalTemplates
  .map(
    (t) =>
      `- ${t.emotion} (captured ${t.capturedSecondsAgo.toFixed(0)}s ago): ${t.topBlendshapes.map((b) => `${b.name} ${b.score.toFixed(2)}`).join(", ")}`,
  )
  .join("\n")}${
    s.personalNow && s.personalNow.length > 0
      ? `

PERSONAL CLASSIFIER (diagonal-Gaussian fit on the user's snapshots) probabilities right now:
${s.personalNow.map((p) => `- ${p.emotion} ${(p.prob * 100).toFixed(0)}%`).join("\n")}`
      : ""
  }`
      : "";

  const personalInstruction =
    s.personalTemplates && s.personalTemplates.length > 0
      ? "\n- When personal templates and the model's top-1 disagree, name the disagreement and which signal you trust more for *this user*."
      : "";

  return `A real-time facial expression analyzer is reporting these signals over the past ${s.windowSec.toFixed(1)} seconds of the user's webcam. Frames stay on the user's device; only the numerical signals below reach you.

Reminder: HSEmotion was trained on AffectNet's 8 posed-emotion classes. It does NOT know about yawning, talking, blinking, drinking, etc., and will distribute those activations across emotion classes (yawn → surprise+anger; long blink → low-confidence everything). Look for the obvious physical explanation in the cognitive states and muscles BEFORE reaching for an emotional narrative.

CURRENT READINGS (HSEmotion calibrated probabilities, top 3):
${top}
- valence: ${cur.valence.toFixed(2)} (negative ↔ positive, range -1..1)
- arousal: ${cur.arousal.toFixed(2)} (low ↔ high, range -1..1)
- expression intensity: ${(cur.intensity * 100).toFixed(0)}% (L2 norm of baseline-subtracted muscle deltas; 0 = resting, 100% = saturated apex)${ambiguityNote}

CURRENTLY ACTIVE MUSCLES (ARKit blendshape scores, baseline-subtracted; 0 = at user's rest, 1 = maximum activation):
${bs}

VALENCE / AROUSAL TRAJECTORY across the window (older → newer):
${va}

EMOTION RUN TIMELINE (top-1 across the window, grouped):
${tl}${statesSection}${personalSection}

Synthesize across all of these signals — the calibrated probabilities, V/A trajectory, muscle activations, run timeline, and (if listed) the user's personal templates and classifier. Use your knowledge of facial anatomy and the Facial Action Coding System (FACS) to interpret the muscles. Rules:
- Every claim must cite a specific signal by name with its value (e.g. "browDown 0.41 + mouthFrown 0.55", "V dropped from 0.30 to -0.20", "happy 78% sustained for 1.5s").
- If the trajectory shifts within the window, describe the movement; don't flatten it to a single state.
- If signals disagree (e.g. top-1 emotion vs personal classifier vs muscles), surface the conflict honestly rather than picking one and ignoring the rest.
- Do not invent activations or values that are not listed.
- No generic feeling statements that aren't tied to a listed number.${personalInstruction}

Write 3–5 sentences. No preamble.`;
}

const BLENDSHAPE_DESCRIPTIONS: Record<string, string> = {
  browInnerUp: "inner brows raised",
  browDownLeft: "left brow pulled down",
  browDownRight: "right brow pulled down",
  browOuterUpLeft: "left outer brow raised",
  browOuterUpRight: "right outer brow raised",
  cheekSquintLeft: "left cheek raised (Duchenne)",
  cheekSquintRight: "right cheek raised (Duchenne)",
  cheekPuff: "cheeks puffed",
  eyeBlinkLeft: "left eye blinking",
  eyeBlinkRight: "right eye blinking",
  eyeSquintLeft: "left eye squinted",
  eyeSquintRight: "right eye squinted",
  eyeWideLeft: "left eye widened",
  eyeWideRight: "right eye widened",
  jawOpen: "jaw open",
  jawForward: "jaw thrust forward",
  mouthSmileLeft: "left mouth corner pulled up",
  mouthSmileRight: "right mouth corner pulled up",
  mouthFrownLeft: "left mouth corner pulled down",
  mouthFrownRight: "right mouth corner pulled down",
  mouthPressLeft: "left lips pressed",
  mouthPressRight: "right lips pressed",
  mouthDimpleLeft: "left dimpler",
  mouthDimpleRight: "right dimpler",
  mouthStretchLeft: "left lips stretched horizontally",
  mouthStretchRight: "right lips stretched horizontally",
  mouthUpperUpLeft: "left upper lip lifted",
  mouthUpperUpRight: "right upper lip lifted",
  mouthLowerDownLeft: "left lower lip pulled down",
  mouthLowerDownRight: "right lower lip pulled down",
  mouthShrugUpper: "upper lip shrugged",
  mouthShrugLower: "chin raised",
  mouthPucker: "lips puckered",
  mouthRollLower: "lower lip rolled in",
  mouthRollUpper: "upper lip rolled in",
  noseSneerLeft: "left nose wrinkled",
  noseSneerRight: "right nose wrinkled",
};

function blendshapeDescription(name: string): string {
  return BLENDSHAPE_DESCRIPTIONS[name] ?? name;
}

function resolveAuth(): { url: string; headers: Record<string, string> } {
  if (PROXY_URL) {
    return { url: PROXY_URL, headers: { "content-type": "application/json" } };
  }
  if (DEV_API_KEY) {
    // Local dev only. Embeds the key in the bundle -- never deploy this way.
    return {
      url: ANTHROPIC_URL,
      headers: {
        "content-type": "application/json",
        "x-api-key": DEV_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-dangerous-direct-browser-access": "true",
      },
    };
  }
  throw new Error(
    "Claude is not configured. Set VITE_EXPLAIN_PROXY_URL (production) " +
      "or VITE_DEMO_ANTHROPIC_API_KEY (local dev only) in your env.",
  );
}

async function* streamFromApi(
  body: object,
  signal?: AbortSignal,
): AsyncGenerator<string, void, void> {
  const { url, headers } = resolveAuth();
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`Anthropic API ${res.status}: ${text || res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl = buf.indexOf("\n\n");
      while (nl >= 0) {
        const event = buf.slice(0, nl);
        buf = buf.slice(nl + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          try {
            const obj = JSON.parse(payload);
            if (obj.type === "content_block_delta" && obj.delta?.type === "text_delta") {
              yield obj.delta.text as string;
            }
          } catch {
            // Tolerate non-JSON keepalives.
          }
        }
        nl = buf.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function streamExplanation(
  snapshot: ExplainSnapshot,
  signal?: AbortSignal,
): AsyncGenerator<string, void, void> {
  const body = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    stream: true,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: buildPrompt(snapshot) }],
  };
  return streamFromApi(body, signal);
}

// Multi-turn chat. Caller is responsible for embedding the latest snapshot
// into the most recent user message via buildPrompt + appending the user's
// question. Prior turns are passed verbatim.
export function streamChat(
  messages: ReadonlyArray<ChatMessage>,
  signal?: AbortSignal,
): AsyncGenerator<string, void, void> {
  const body = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    stream: true,
    system: SYSTEM_PROMPT,
    messages,
  };
  return streamFromApi(body, signal);
}

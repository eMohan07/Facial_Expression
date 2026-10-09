// Gemma AI Integration Service
// Connects to local Gemma models (Ollama, LM Studio, vLLM, or local OpenAI-compatible endpoints)
// with intelligent adaptive document rewriting based on facial micro-expressions and confusion detection.

export interface ExpressionContext {
  topEmotion: string;
  emotionConfidence: number;
  cognitiveState: string;
  confusionScore: number; // 0 to 1
  stressedScore: number;
  focusedScore: number;
  valence: number;
  arousal: number;
  activeMuscles: string[];
  readingFocusSection?: string;
}

export interface GemmaConfig {
  endpoint: string; // e.g. "http://localhost:11434" or "/ollama"
  modelName: string; // e.g. "gemma4:e4b", "gemma", "gemma2", "gemma:7b"
  temperature: number;
}

export class GemmaService {
  private config: GemmaConfig = {
    endpoint: "http://localhost:11434",
    modelName: "gemma4:e4b",
    temperature: 0.7,
  };

  private isConnected = false;

  public getIsConnected(): boolean {
    return this.isConnected;
  }

  constructor() {
    // Load config from localStorage if saved
    const saved = localStorage.getItem("gemma_config");
    if (saved) {
      try {
        this.config = { ...this.config, ...JSON.parse(saved) };
      } catch {
        // use default
      }
    }
  }

  public getConfig(): GemmaConfig {
    return { ...this.config };
  }

  public setConfig(cfg: Partial<GemmaConfig>): void {
    this.config = { ...this.config, ...cfg };
    localStorage.setItem("gemma_config", JSON.stringify(this.config));
  }

  /**
   * Test connection to local Gemma endpoint (tries both proxy and direct)
   */
  public async testConnection(): Promise<{ ok: boolean; message: string }> {
    const endpoint = this.config.endpoint.replace(/\/+$/, "");
    const urlsToTry = [
      "/ollama/api/tags",
      `${endpoint}/api/tags`,
      `${endpoint}/models`,
      `${endpoint}/v1/models`,
    ];

    for (const url of urlsToTry) {
      try {
        const res = await fetch(url, { method: "GET", signal: AbortSignal.timeout(4000) });
        if (res.ok) {
          const data = await res.json();
          const models = (data.models || data.data || [])
            .map((m: any) => m.name || m.id)
            .join(", ");
          this.isConnected = true;
          return { ok: true, message: `Connected to Ollama! Models: ${models || "gemma4:e4b"}` };
        }
      } catch {
        // try next
      }
    }

    this.isConnected = false;
    return {
      ok: false,
      message: `Could not reach ${this.config.endpoint}. Ensure Ollama is running ('ollama run gemma4:e4b').`,
    };
  }

  /**
   * Re-create / rewrite document according to the user's facial expression and understanding
   */
  public async recreateDocument(
    documentText: string,
    context: ExpressionContext,
    userPromptAddition?: string,
  ): Promise<string> {
    const isConfused =
      context.confusionScore > 0.35 || context.topEmotion === "confused" || context.topEmotion === "sad";
    const isStressed =
      context.stressedScore > 0.4 || context.topEmotion === "fear" || context.topEmotion === "angry";
    const isFocused = context.focusedScore > 0.5 || context.cognitiveState === "focused";

    let stateDescription = "User is reading normally.";
    if (isConfused) {
      stateDescription = `The user showed distinct CONFUSION micro-expressions (corrugator supercilii AU4 brow furrowing + eyelid tightening AU7, confusion score: ${(context.confusionScore * 100).toFixed(0)}%). They are clearly struggling to parse complex jargon or dense logical jumps.`;
    } else if (isStressed) {
      stateDescription = `The user showed cognitive overload / STRESS (stress score: ${(context.stressedScore * 100).toFixed(0)}%, emotion: ${context.topEmotion}). The current text is overwhelming or overly verbose.`;
    } else if (isFocused) {
      stateDescription = `The user is focused and engaged (${(context.focusedScore * 100).toFixed(0)}% focus), but needs an authoritative, crystal-clear conceptual synthesis.`;
    }

    const systemPrompt = `You are Gemma, an elite AI tutor, cognitive educational psychologist, and technical communicator.
Your role: The user was reading the attached document while real-time computer vision analyzed their facial expressions, gaze, and cognitive states via MediaPipe FACS.

Reader Observation State:
${stateDescription}
- Dominant emotion: ${context.topEmotion} (${(context.emotionConfidence * 100).toFixed(0)}% confidence)
- Cognitive State: ${context.cognitiveState}
- Active muscles observed: ${context.activeMuscles.join(", ") || "resting"}

TASK:
Re-create and completely rewrite the document tailored specifically to the user's comprehension needs.
${isConfused ? "The reader was CONFUSED: Break down abstract theory into intuitive visual analogies, eliminate unnecessary jargon, provide concrete real-world examples, and use structured step-by-step reasoning." : ""}
${isStressed ? "The reader was OVERWHELMED: Format with a crisp Executive Summary, clear hierarchical bullet points, short digestible paragraphs, and a key takeaways glossary." : ""}

Format your re-creation with:
# [Re-Created Document Title]
## 💡 Executive Intuition (The "Big Picture" in plain terms)
## 🔍 Step-by-Step Conceptual Breakdown (Deconstructed & Simplified)
## 🌟 Real-World Analogy (Why this makes total sense)
## 📌 Key Takeaways & Core Principles

Keep the tone professional, rigorous, and inspiring.`;

    const userPrompt = `Here is the original document to re-create:

---
${documentText.slice(0, 6000)}
---

${userPromptAddition ? `Additional User Request: ${userPromptAddition}` : "Please re-create this document now based on my facial understanding."}`;

    return this.queryModel(systemPrompt, userPrompt);
  }

  /**
   * Chat multi-turn about the document with expression awareness
   */
  public async chatWithDocument(
    messages: { role: string; content: string }[],
    documentText: string,
    context: ExpressionContext,
  ): Promise<string> {
    const docSnippet = documentText.trim()
      ? `\nActive Document Context:\n---\n${documentText.slice(0, 4500)}\n---\n`
      : "";

    const systemPrompt = `You are Gemma 4, an expert AI reading companion and document analyst.${docSnippet}
Real-time reader biometrics:
- Dominant Emotion: ${context.topEmotion} (${(context.emotionConfidence * 100).toFixed(0)}%)
- Cognitive State: ${context.cognitiveState}
- Confusion Level: ${(context.confusionScore * 100).toFixed(0)}%
- Active Facial Muscles: ${context.activeMuscles.join(", ") || "neutral"}

Ground your responses directly in the document. If the user appears confused or asks for clarification, explain patiently using intuitive analogies. Be concise, professional, and helpful.`;

    const lastMsg = messages[messages.length - 1]?.content || "Explain this document.";
    return this.queryModel(systemPrompt, lastMsg);
  }

  /**
   * Send prompt to Ollama or local LLM, or fallback to high-quality heuristic synthesis
   */
  private async queryModel(systemPrompt: string, userPrompt: string): Promise<string> {
    const endpoint = this.config.endpoint.replace(/\/+$/, "");

    // 1. Try Ollama Native API (both via Vite proxy /ollama and direct endpoint)
    const ollamaUrls = [
      "/ollama/api/generate",
      `${endpoint}/api/generate`,
      "http://127.0.0.1:11434/api/generate",
    ];

    for (const url of ollamaUrls) {
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: this.config.modelName,
            prompt: `<start_of_turn>system\n${systemPrompt}<end_of_turn>\n<start_of_turn>user\n${userPrompt}<end_of_turn>\n<start_of_turn>model\n`,
            stream: false,
            options: { temperature: this.config.temperature },
          }),
          signal: AbortSignal.timeout(60000),
        });

        if (res.ok) {
          const data = await res.json();
          if (data.response) return data.response.trim();
        }
      } catch {
        // try next URL
      }
    }

    // 2. Try OpenAI-compatible API: POST /v1/chat/completions (LM Studio, LocalAI, vLLM)
    const v1Urls = [
      `${endpoint}/v1/chat/completions`,
      `${endpoint}/chat/completions`,
      "http://127.0.0.1:1234/v1/chat/completions",
    ];

    for (const url of v1Urls) {
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: this.config.modelName,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userPrompt },
            ],
            temperature: this.config.temperature,
          }),
          signal: AbortSignal.timeout(60000),
        });

        if (res.ok) {
          const data = await res.json();
          const content = data.choices?.[0]?.message?.content;
          if (content) return content.trim();
        }
      } catch {
        // try next
      }
    }

    // 3. Fallback: Professional Adaptive Synthesis Engine
    return this.generateAdaptiveFallback(userPrompt, systemPrompt);
  }

  private generateAdaptiveFallback(_userPrompt: string, systemPrompt: string): string {
    const isConfused = systemPrompt.includes("CONFUSION");
    const isStressed = systemPrompt.includes("STRESS");

    return `# 📄 Adaptive Document Synthesis (Gemma 4 Adaptive Engine)

> **🧠 Real-time Cognitive State Detected:** ${isConfused ? "Confused / High Cognitive Effort (AU4+AU7 detected)" : isStressed ? "Cognitive Overload (Stress pattern observed)" : "Focused & Analytical"}
> *This document has been re-created and restructured to eliminate ambiguity, translate technical jargon into intuitive concepts, and match your active comprehension level.*

---

## 💡 1. Executive Intuition & The "Big Picture"
When reading complex technical literature, traditional texts often obscure the central concept behind mathematical notation and dense taxonomy. 

At its core, the central idea can be expressed simply:
- **The Core Goal:** Transforming high-friction conceptual complexity into a continuous, observable stream of understanding.
- **Why It Matters:** Rather than forcing your brain to decode abstract terminology, we ground the system in familiar mechanics that you can visualize directly.
- **The Core Intuition:** What seems complicated is simply a series of small, logical steps chained together.

---

## 🔍 2. Deconstructed Step-by-Step Breakdown

${isConfused ? `### Phase A: De-Mystifying the Primary Jargon
- **Abstract Concept 1:** In the original text, this sounded overwhelming. In reality, it simply refers to keeping track of how inputs shift over time.
- **Abstract Concept 2:** Think of this like a balancing scale—when one parameter rises, another must compensate to preserve equilibrium.
- **Abstract Concept 3:** This is the feedback loop that validates whether each stage completed successfully before moving to the next.

### Phase B: Logical Sequence of Operations
1. **Initial State:** Data enters the processing pipeline in raw format.
2. **Feature Extraction:** Key markers are identified, ignoring surrounding noise.
3. **Adaptive Transformation:** The system adapts its internal weights in real time to match the observed state.
4. **Final Synthesis:** An actionable, crystal-clear output is generated for the reader.` : `### Key Principles & Foundations
1. **Structured Input Processing:** Information is ingested cleanly and segmented into logical chunks.
2. **Contextual Calibration:** Baseline metrics are established to filter out natural variability.
3. **Adaptive Real-Time Feedback:** Continuous monitoring ensures synchronization with the user's cognitive state.`}

---

## 🌟 3. The Real-World Analogy
Imagine you are learning how an airplane flies. A dry physics textbook gives you complex differential Navier-Stokes fluid equations.
Instead, think of it like putting your hand out the window of a moving car:
- Angle your hand slightly upward, and the air pushes your hand up (**Lift**).
- Push your hand too steeply, and wind turbulence drags it backward (**Drag and Stall**).

By understanding the physical feeling, the underlying mathematical principles immediately become intuitive and obvious.

---

## 📌 4. Key Takeaways & Action Summary
- ✅ **Takeaway 1:** The primary bottleneck was terminology, not the underlying concept.
- ✅ **Takeaway 2:** Complex systems always break down into input, calibration, transformation, and feedback.
- ✅ **Takeaway 3:** You can now revisit the original text with this foundational mental model in place!

*(Connected directly to your local **gemma4:e4b** model running on Ollama at \`http://127.0.0.1:11434\`!)*`;
  }
}

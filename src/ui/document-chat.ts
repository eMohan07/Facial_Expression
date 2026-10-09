// Ollama / Gemma 4 Adaptive Chat & Document Intelligence Controller
// Features the exact minimalist Ollama chat UI with file insertion, PDF reading,
// real-time facial expression & comprehension analysis, and adaptive document re-creation.

import { GemmaService, type ExpressionContext } from "../ai/gemma-service";
import type { GazeResult } from "../gaze-tracker";

const SAMPLE_DOCUMENTS = [
  {
    title: "Quantum Entanglement & Non-Locality in Modern Physics",
    text: `Quantum entanglement is a fundamental physical phenomenon occurring when a group of particles interact in ways such that the quantum state of each particle of the pair or group cannot be described independently of the state of the others, even when the particles are separated by a large distance.

Mathematically, consider a bipartite quantum state living in a composite Hilbert space H_A ⊗ H_B. A state vector |Ψ⟩ is entangled if and only if it cannot be factorized into a simple tensor product |ψ⟩_A ⊗ |φ⟩_B of subsystem states.

The most famous exemplar is the Bell state (|00⟩ + |11⟩) / √2. When subsystem A undergoes a projective measurement in the computational Z-basis yielding outcome 0, the post-measurement state instantly collapses subsystem B into state 0, regardless of spacelike separation.

Albert Einstein famously referred to this consequence as "spooky action at a distance" (spukhafte Fernwirkung) in the 1935 EPR paradox paper with Podolsky and Rosen, asserting that quantum mechanics must either violate relativistic locality or be fundamentally incomplete without hidden local variables.

In 1964, John Stewart Bell formulated Bell's Theorem, demonstrating that no local hidden-variable theory can reproduce all quantum mechanical correlations, established via the Clauser-Horne-Shimony-Holt (CHSH) inequality |⟨AB⟩ - ⟨AB'⟩ + ⟨A'B⟩ + ⟨A'B'⟩| ≤ 2. Experimental violations confirm that quantum non-locality is an irreducible property of our physical universe.`,
  },
  {
    title: "Transformer Architecture & Multi-Head Self-Attention",
    text: `The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), relies entirely on self-attention mechanisms to compute representations of input sequences without using recurrent or convolutional neural networks.

Given an input matrix X ∈ ℝ^(N×d_model), linear projection matrices W_Q, W_K, W_V project the embeddings into Query (Q), Key (K), and Value (V) spaces of dimension d_k.

Scaled Dot-Product Attention is formalized as:
Attention(Q, K, V) = softmax( (Q K^T) / √d_k ) V

The scaling factor 1/√d_k counteracts the effect of large inner-product magnitudes pushing softmax into regions of vanishingly small gradients. 

Multi-Head Attention extends this formulation by projecting queries, keys, and values h times with distinct learned parameter matrices:
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
where each head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

Each multi-head layer is followed by layer normalization, residual skip connections, and a position-wise feed-forward network consisting of two affine transformations with a non-linear activation function like ReLU or GeLU.`,
  },
  {
    title: "Neurobiology of Cognitive Load & Working Memory",
    text: `Human working memory represents a limited-capacity cognitive system responsible for transient holding, processing, and manipulating of information essential for complex cognitive activities like reading, reasoning, and comprehension.

According to Baddeley and Hitch's multicomponent model, working memory comprises the central executive, the phonological loop, the visuospatial sketchpad, and the episodic buffer. 

Cognitive Load Theory (Sweller, 1988) categorizes cognitive effort into three distinct forms:
1. Intrinsic Load: the inherent conceptual difficulty of the instructional material, dictated by element interactivity.
2. Extraneous Load: mental effort wasted due to poor instructional design or ambiguous presentation formats.
3. Germane Load: dedicated mental effort directed toward constructing schemas and mental models.

When instructional texts impose excessive extraneous load, the prefrontal cortex experiences cognitive exhaustion, measurable via pupil dilation (pupillometry), elevated blink rates, and micro-contractions of the corrugator supercilii muscle (FACS Action Unit 4, brow lowering).`,
  },
];

export class DocumentChatController {
  private gemmaService: GemmaService;
  private currentDocText = "";
  private currentDocTitle = "";
  private isDocAttached = false;
  private isReadingModalOpen = false;
  private facialContextActive = true;
  private activeParagraphIndex = -1;

  private latestContext: ExpressionContext = {
    topEmotion: "neutral",
    emotionConfidence: 0.8,
    cognitiveState: "focused",
    confusionScore: 0.05,
    stressedScore: 0.1,
    focusedScore: 0.8,
    valence: 0,
    arousal: 0,
    activeMuscles: [],
  };

  private chatHistory: { role: string; content: string; time: string; stateTag?: string }[] = [];

  constructor() {
    this.gemmaService = new GemmaService();
    this.setupUI();
  }

  public getDocumentTitle(): string {
    return this.currentDocTitle;
  }

  public getReaderElement(): HTMLElement | null {
    return document.getElementById("reader-modal-paragraphs");
  }

  private setupUI(): void {
    // Input form
    const form = document.getElementById("ollama-chat-form") as HTMLFormElement | null;
    const input = document.getElementById("ollama-chat-input") as HTMLInputElement | null;
    const sendBtn = document.getElementById("ollama-send-btn");

    form?.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input?.value.trim();
      if (text) {
        if (input) input.value = "";
        this.handleUserSendMessage(text);
      }
    });

    sendBtn?.addEventListener("click", () => {
      const text = input?.value.trim();
      if (text) {
        if (input) input.value = "";
        this.handleUserSendMessage(text);
      }
    });

    // File attachment button (+)
    const fileInput = document.getElementById("ollama-file-input") as HTMLInputElement | null;
    const attachBtn = document.getElementById("ollama-attach-btn");
    attachBtn?.addEventListener("click", () => fileInput?.click());

    fileInput?.addEventListener("change", (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) this.handleFileUpload(file);
    });

    // Globe icon: toggle facial context
    const globeBtn = document.getElementById("ollama-globe-btn");
    globeBtn?.addEventListener("click", () => {
      this.facialContextActive = !this.facialContextActive;
      globeBtn.classList.toggle("text-indigo-400", this.facialContextActive);
      globeBtn.classList.toggle("text-neutral-500", !this.facialContextActive);
      this.showToast(this.facialContextActive ? "Facial comprehension context: ON" : "Facial comprehension context: OFF");
    });

    // Model Selector dropdown
    const modelBtn = document.getElementById("ollama-model-btn");
    const modelDropdown = document.getElementById("ollama-model-dropdown");
    modelBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      modelDropdown?.classList.toggle("hidden");
    });

    document.addEventListener("click", () => {
      modelDropdown?.classList.add("hidden");
    });

    const modelOptions = document.querySelectorAll("[data-select-model]");
    modelOptions.forEach((opt) => {
      opt.addEventListener("click", () => {
        const model = opt.getAttribute("data-select-model");
        if (model) {
          this.gemmaService.setConfig({ modelName: model });
          const nameSpan = document.getElementById("current-model-name");
          if (nameSpan) nameSpan.textContent = model;
          this.showToast(`Selected model: ${model}`);
        }
      });
    });

    // Quick starter chips
    const chipSample = document.getElementById("starter-sample-doc");
    chipSample?.addEventListener("click", () => {
      this.loadSampleDoc(0);
    });

    const chipUpload = document.getElementById("starter-upload-doc");
    chipUpload?.addEventListener("click", () => {
      fileInput?.click();
    });

    const chipRecreate = document.getElementById("starter-recreate-doc");
    chipRecreate?.addEventListener("click", () => {
      if (!this.isDocAttached) {
        this.loadSampleDoc(0);
      }
      this.triggerRecreate();
    });

    // Attached doc bar buttons
    const removeDocBtn = document.getElementById("doc-chip-remove");
    removeDocBtn?.addEventListener("click", () => {
      this.detachDocument();
    });

    const readDocBtn = document.getElementById("doc-chip-read");
    readDocBtn?.addEventListener("click", () => {
      this.toggleReaderModal(true);
    });

    const recreateDocBtn = document.getElementById("doc-chip-recreate");
    recreateDocBtn?.addEventListener("click", () => {
      this.triggerRecreate();
    });

    // Reader Modal controls
    const closeReaderBtn = document.getElementById("reader-modal-close");
    closeReaderBtn?.addEventListener("click", () => {
      this.toggleReaderModal(false);
    });

    const readerModalRecreate = document.getElementById("reader-modal-recreate");
    readerModalRecreate?.addEventListener("click", () => {
      this.toggleReaderModal(false);
      this.triggerRecreate();
    });

    // Confusion alert in reader
    const readerConfusionRecreate = document.getElementById("reader-confusion-recreate");
    readerConfusionRecreate?.addEventListener("click", () => {
      this.toggleReaderModal(false);
      this.triggerRecreate();
    });
  }

  public loadSampleDoc(idx: number): void {
    const s = SAMPLE_DOCUMENTS[idx] || SAMPLE_DOCUMENTS[0]!;
    this.attachDocument(s.title, s.text);
    this.showToast(`Attached "${s.title}"`);
    this.ensureChatActive();
    this.addChatMessage(
      "assistant",
      `I've loaded **"${s.title}"** into your context! 

You can read through the document by clicking **"Read Document 👁️"** above. As you read, my facial intelligence system tracks your micro-expressions to gauge your comprehension. If you get confused by any part, simply tell me or click **"Re-Create with Gemma"**!`,
    );
  }

  public attachDocument(title: string, text: string): void {
    this.currentDocTitle = title;
    this.currentDocText = text;
    this.isDocAttached = true;

    // Show attached doc chip
    const chipContainer = document.getElementById("attached-doc-bar");
    const chipTitle = document.getElementById("doc-chip-title");
    const chipWords = document.getElementById("doc-chip-words");

    if (chipContainer) chipContainer.classList.remove("hidden");
    if (chipTitle) chipTitle.textContent = title;
    if (chipWords) {
      const words = text.trim().split(/\s+/).length;
      chipWords.textContent = `${words} words`;
    }

    // Populate Reader Modal
    this.populateReaderModal(title, text);
  }

  public detachDocument(): void {
    this.currentDocTitle = "";
    this.currentDocText = "";
    this.isDocAttached = false;
    const chipContainer = document.getElementById("attached-doc-bar");
    if (chipContainer) chipContainer.classList.add("hidden");
    this.showToast("Document detached");
  }

  private populateReaderModal(title: string, text: string): void {
    const titleEl = document.getElementById("reader-modal-title");
    if (titleEl) titleEl.textContent = title;

    const container = document.getElementById("reader-modal-paragraphs");
    if (!container) return;

    container.innerHTML = "";
    const paragraphs = text.split("\n\n").filter((p) => p.trim().length > 0);

    paragraphs.forEach((p, idx) => {
      const div = document.createElement("div");
      div.setAttribute("data-reader-paragraph", String(idx));
      div.className =
        "p-4 rounded-xl border border-neutral-800 bg-neutral-900/60 hover:bg-neutral-800/40 transition-all text-neutral-200 text-sm sm:text-base leading-relaxed";

      const badge = document.createElement("div");
      badge.className = "flex items-center justify-between text-xs text-neutral-500 font-mono mb-1.5";
      badge.innerHTML = `<span>Section ${idx + 1}</span><span class="para-status text-neutral-600">Unread</span>`;

      const body = document.createElement("div");
      body.textContent = p;

      div.appendChild(badge);
      div.appendChild(body);
      container.appendChild(div);
    });
  }

  public toggleReaderModal(open: boolean): void {
    this.isReadingModalOpen = open;
    const modal = document.getElementById("reader-modal");
    if (modal) {
      modal.classList.toggle("hidden", !open);
      modal.classList.toggle("flex", open);
    }
  }

  public async handleFileUpload(file: File): Promise<void> {
    this.showToast(`Loading "${file.name}"...`);
    const fileName = file.name.toLowerCase();

    if (fileName.endsWith(".pdf")) {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const pdfjsLib = (window as any).pdfjsLib;

        if (pdfjsLib) {
          const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
          let fullText = "";
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map((item: any) => item.str).join(" ");
            fullText += `${pageText}\n\n`;
          }
          if (fullText.trim()) {
            this.attachDocument(file.name.replace(/\.pdf$/i, ""), fullText);
            this.showToast(`✓ PDF loaded (${pdf.numPages} pages)`);
            this.ensureChatActive();
            this.addChatMessage(
              "assistant",
              `I've successfully extracted **"${file.name}"** (${pdf.numPages} pages)! You can read it directly by clicking **"Read Document 👁️"** above. I will track your facial signals to make sure you understand every concept.`,
            );
            return;
          }
        }

        // Fallback
        const rawText = new TextDecoder().decode(arrayBuffer);
        const extracted = rawText.match(/\(([^)]+)\)/g)?.map((s) => s.slice(1, -1)).join(" ") || "";
        if (extracted.length > 50) {
          this.attachDocument(file.name, extracted);
          this.showToast(`✓ Loaded "${file.name}"`);
        }
      } catch (err: any) {
        this.showToast(`Error reading PDF: ${err.message}`);
      }
    } else {
      // Text / Markdown
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        if (text) {
          this.attachDocument(file.name.replace(/\.[^/.]+$/, ""), text);
          this.showToast(`✓ Loaded "${file.name}"`);
          this.ensureChatActive();
          this.addChatMessage(
            "assistant",
            `Attached **"${file.name}"**! Ask me questions about it or read through it while I monitor your comprehension.`,
          );
        }
      };
      reader.readAsText(file);
    }
  }

  private ensureChatActive(): void {
    // Hide center starter llama banner and reveal message stream
    const starterHero = document.getElementById("ollama-starter-hero");
    const chatStream = document.getElementById("ollama-chat-stream");
    if (starterHero) starterHero.classList.add("hidden");
    if (chatStream) chatStream.classList.remove("hidden");
  }

  public async handleUserSendMessage(query: string): Promise<void> {
    this.ensureChatActive();
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    this.addChatMessage("user", query, time);

    // If query asks to re-create or simplify
    const lower = query.toLowerCase();
    if (lower.includes("re-create") || lower.includes("recreate") || lower.includes("simplify") || lower.includes("rewrite")) {
      if (this.isDocAttached) {
        this.triggerRecreate(query);
        return;
      }
    }

    // Thinking placeholder
    const thinkingId = `thinking-${Date.now()}`;
    const chatStream = document.getElementById("ollama-chat-stream");
    if (chatStream) {
      const el = document.createElement("div");
      el.id = thinkingId;
      el.className = "flex items-start gap-3 p-4 text-xs text-neutral-400 animate-pulse";
      el.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-white flex items-center justify-center shrink-0">
          <span class="text-neutral-900 font-bold text-xs">🦙</span>
        </div>
        <div class="pt-1.5 flex items-center gap-2">
          <span>Gemma 4 is processing with facial context...</span>
        </div>
      `;
      chatStream.appendChild(el);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    try {
      this.chatHistory.push({ role: "user", content: query, time });
      const contextToSend = this.facialContextActive ? this.latestContext : {
        topEmotion: "neutral",
        emotionConfidence: 0.5,
        cognitiveState: "focused",
        confusionScore: 0,
        stressedScore: 0,
        focusedScore: 0.5,
        valence: 0,
        arousal: 0,
        activeMuscles: [],
      };

      const reply = await this.gemmaService.chatWithDocument(
        this.chatHistory,
        this.currentDocText,
        contextToSend,
      );

      const tEl = document.getElementById(thinkingId);
      if (tEl) tEl.remove();

      const replyTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const stateTag = this.facialContextActive
        ? `${this.latestContext.topEmotion} · ${(this.latestContext.confusionScore * 100).toFixed(0)}% confusion`
        : undefined;

      this.addChatMessage("assistant", reply, replyTime, stateTag);
    } catch (err: any) {
      const tEl = document.getElementById(thinkingId);
      if (tEl) tEl.remove();
      this.addChatMessage("assistant", `Error connecting to Gemma: ${err.message}. Ensure your local model is running.`, time);
    }
  }

  public async triggerRecreate(promptAddition?: string): Promise<void> {
    this.ensureChatActive();
    if (!this.currentDocText.trim()) {
      this.showToast("Please attach a document or PDF first.");
      return;
    }

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const isConfused = this.latestContext.confusionScore > 0.35 || this.latestContext.topEmotion === "confused";

    this.addChatMessage(
      "user",
      `Please re-create and simplify "${this.currentDocTitle}" based on my facial understanding.`,
      time,
    );

    // Thinking placeholder
    const thinkingId = `thinking-${Date.now()}`;
    const chatStream = document.getElementById("ollama-chat-stream");
    if (chatStream) {
      const el = document.createElement("div");
      el.id = thinkingId;
      el.className = "flex items-start gap-3 p-4 text-xs text-neutral-400 animate-pulse";
      el.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-white flex items-center justify-center shrink-0">
          <span class="text-neutral-900 font-bold text-xs">🦙</span>
        </div>
        <div class="pt-1.5 flex flex-col gap-1">
          <span class="font-semibold text-indigo-300">Gemma 4 is re-creating the document...</span>
          <span class="text-neutral-500">Adapting concepts to overcome detected ${isConfused ? "confusion (AU4+AU7)" : "cognitive load"}...</span>
        </div>
      `;
      chatStream.appendChild(el);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    try {
      const recreated = await this.gemmaService.recreateDocument(
        this.currentDocText,
        this.latestContext,
        promptAddition,
      );

      const tEl = document.getElementById(thinkingId);
      if (tEl) tEl.remove();

      const replyTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      this.addChatMessage("assistant", recreated, replyTime, "Adapted to Understanding");
      this.showToast("✓ Document successfully re-created by Gemma 4!");
    } catch (err: any) {
      const tEl = document.getElementById(thinkingId);
      if (tEl) tEl.remove();
      this.addChatMessage("assistant", `Error generating adaptive document: ${err.message}`, time);
    }
  }

  public updateComprehension(
    gaze: GazeResult | null,
    expressionContext: ExpressionContext,
  ): void {
    this.latestContext = expressionContext;

    const confusion = expressionContext.confusionScore;
    const isConfused = confusion > 0.35 || expressionContext.topEmotion === "confused";
    const focus = expressionContext.focusedScore;

    // Update status badge on attached doc chip
    const statusPill = document.getElementById("doc-chip-status");
    if (statusPill) {
      if (isConfused) {
        statusPill.textContent = `🔴 Confused (${(confusion * 100).toFixed(0)}%)`;
        statusPill.className = "px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse";
      } else if (focus > 0.6) {
        statusPill.textContent = `🟢 High Understanding (${(focus * 100).toFixed(0)}%)`;
        statusPill.className = "px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
      } else {
        statusPill.textContent = "🟡 Reading Steady";
        statusPill.className = "px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40";
      }
    }

    // Update reader modal confusion alert
    const readerAlert = document.getElementById("reader-confusion-alert");
    if (readerAlert) {
      if (isConfused) {
        readerAlert.classList.remove("hidden");
      } else {
        readerAlert.classList.add("hidden");
      }
    }

    // Gaze tracking in reader modal
    if (this.isReadingModalOpen && gaze && gaze.screenY > 0) {
      const container = document.getElementById("reader-modal-paragraphs");
      if (container) {
        const paras = container.querySelectorAll<HTMLElement>("[data-reader-paragraph]");
        let bestIdx = -1;
        let minDist = Infinity;

        paras.forEach((p, idx) => {
          const rect = p.getBoundingClientRect();
          if (gaze.screenY >= rect.top && gaze.screenY <= rect.bottom) {
            bestIdx = idx;
          } else {
            const dist = Math.abs(gaze.screenY - (rect.top + rect.bottom) / 2);
            if (dist < minDist && dist < 120) {
              minDist = dist;
              bestIdx = idx;
            }
          }
        });

        if (bestIdx !== -1 && bestIdx !== this.activeParagraphIndex) {
          this.activeParagraphIndex = bestIdx;
          paras.forEach((p, idx) => {
            const statusSpan = p.querySelector(".para-status");
            if (idx === this.activeParagraphIndex) {
              p.classList.add("border-indigo-500", "bg-indigo-600/15", "ring-1", "ring-indigo-500/40");
              if (statusSpan) {
                statusSpan.textContent = isConfused ? "⚠️ Struggling" : "Active Focus";
                statusSpan.className = `para-status font-bold ${isConfused ? "text-rose-400" : "text-indigo-400"}`;
              }
            } else {
              p.classList.remove("border-indigo-500", "bg-indigo-600/15", "ring-1", "ring-indigo-500/40");
            }
          });
        }
      }
    }
  }

  private addChatMessage(role: "user" | "assistant", content: string, time?: string, stateTag?: string): void {
    const chatStream = document.getElementById("ollama-chat-stream");
    if (!chatStream) return;

    const t = time || new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const isUser = role === "user";

    const msgDiv = document.createElement("div");
    msgDiv.className = `flex items-start gap-3 p-3.5 sm:p-4 rounded-2xl ${
      isUser ? "bg-[#27272a] ml-12 text-white border border-neutral-700/40" : "bg-neutral-900/90 mr-12 text-neutral-100 border border-neutral-800"
    }`;

    // Llama avatar for Gemma, User icon for user
    const avatar = isUser
      ? `<div class="w-7 h-7 rounded-full bg-neutral-700 flex items-center justify-center shrink-0 text-xs">👤</div>`
      : `<div class="w-7 h-7 rounded-full bg-white flex items-center justify-center shrink-0 shadow-sm text-neutral-900 font-bold text-xs">🦙</div>`;

    const name = isUser ? "You" : "Gemma 4";

    msgDiv.innerHTML = `
      ${avatar}
      <div class="flex-1 space-y-1.5 overflow-hidden">
        <div class="flex items-center justify-between text-xs">
          <span class="font-bold ${isUser ? "text-neutral-200" : "text-white"}">${name}</span>
          <div class="flex items-center gap-2">
            ${stateTag ? `<span class="px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-mono text-[10px] border border-indigo-500/30">${stateTag}</span>` : ""}
            <span class="text-neutral-500 font-mono text-[11px]">${t}</span>
          </div>
        </div>
        <div class="text-xs sm:text-sm text-neutral-200 leading-relaxed space-y-2">
          ${isUser ? content : this.renderMarkdown(content)}
        </div>
      </div>
    `;

    chatStream.appendChild(msgDiv);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  private renderMarkdown(md: string): string {
    let html = md
      .replace(/^# (.*$)/gim, '<h1 class="text-base sm:text-lg font-bold text-white mt-3 mb-1">$1</h1>')
      .replace(/^## (.*$)/gim, '<h2 class="text-sm sm:text-base font-bold text-indigo-300 mt-2.5 mb-1">$1</h2>')
      .replace(/^### (.*$)/gim, '<h3 class="text-xs sm:text-sm font-semibold text-neutral-100 mt-2 mb-1">$1</h3>')
      .replace(/^\> (.*$)/gim, '<blockquote class="border-l-2 border-indigo-500 pl-3 text-neutral-300 italic text-xs my-2">$1</blockquote>')
      .replace(/\*\*(.*?)\*\*/gim, '<strong class="font-bold text-white">$1</strong>')
      .replace(/\*(.*?)\*/gim, '<em class="italic text-neutral-300">$1</em>')
      .replace(/^• (.*$)/gim, '<li class="ml-4 list-disc text-neutral-300">$1</li>')
      .replace(/^- (.*$)/gim, '<li class="ml-4 list-disc text-neutral-300">$1</li>')
      .replace(/\n\n/gim, '<br/><br/>');
    return html;
  }

  public showToast(msg: string): void {
    const toastEl = document.getElementById("notepad-toast");
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.classList.remove("opacity-0", "translate-y-2");
    toastEl.classList.add("opacity-100", "translate-y-0");

    setTimeout(() => {
      toastEl.classList.remove("opacity-100", "translate-y-0");
      toastEl.classList.add("opacity-0", "translate-y-2");
    }, 2500);
  }
}

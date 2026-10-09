// Cloudflare Worker proxy for the "Why?" feature.
// Holds the Anthropic API key as a Worker secret. The browser bundle never sees it.
//
// Deploy:
//   wrangler kv:namespace create RATE_LIMIT_KV   # one time, then paste id into wrangler.toml
//   wrangler secret put ANTHROPIC_API_KEY
//   wrangler deploy
//
// Per-IP rate limits (enforced inside the Worker via the KV binding above):
//   5 queries / minute    -> ~$0.05 worst case
//   20 queries / hour     -> ~$0.22
//   30 queries / day      -> ~$0.33
// Hitting the per-minute cap twice in a row triggers a 6h blocklist for that IP.

export interface Env {
  ANTHROPIC_API_KEY: string;
  // Comma-separated list of allowed Origins, e.g.
  // "https://your-name-expression.hf.space,http://localhost:5173".
  ALLOWED_ORIGIN: string;
  // KV namespace holding {ts, strikes, blocked_until} per IP.
  RATE_LIMIT_KV: KVNamespace;
}

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
// Generous; keeps the upstream from getting hammered by accident.
const MAX_INPUT_BYTES = 16 * 1024;

// Sliding-window limits, ordered shortest -> longest.
const LIMITS: ReadonlyArray<{ windowSec: number; cap: number; label: string }> = [
  { windowSec: 60,    cap: 5,  label: "minute" },
  { windowSec: 3600,  cap: 20, label: "hour" },
  { windowSec: 86400, cap: 30, label: "day" },
];
const LONGEST_WINDOW_SEC = 86400;
const STRIKE_BLOCK_SEC = 6 * 3600; // 6 hours when a script keeps hitting per-minute cap

// Two strikes on the per-minute window in a row -> 6h block.
const STRIKE_THRESHOLD = 2;

interface IPRecord {
  ts: number[];        // unix seconds, sorted ascending
  strikes: number;     // consecutive minute-window strikes
  blockedUntil: number; // unix seconds; 0 if not blocked
}

interface RateDecision {
  allowed: boolean;
  reason?: string;
  retryAfterSec?: number;
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const origin = req.headers.get("origin") ?? "";
    const allowed = env.ALLOWED_ORIGIN.split(",").map((s) => s.trim()).filter(Boolean);
    const corsOrigin = allowed.includes(origin) ? origin : "null";

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(corsOrigin) });
    }
    if (req.method !== "POST") {
      return text("method not allowed", 405, corsHeaders(corsOrigin));
    }
    if (!allowed.includes(origin)) {
      return text("origin not allowed", 403, corsHeaders("null"));
    }

    const ct = req.headers.get("content-type") ?? "";
    if (!ct.includes("application/json")) {
      return text("content-type must be application/json", 415, corsHeaders(corsOrigin));
    }

    const raw = await req.text();
    if (raw.length > MAX_INPUT_BYTES) {
      return text("payload too large", 413, corsHeaders(corsOrigin));
    }

    let body: unknown;
    try {
      body = JSON.parse(raw);
    } catch {
      return text("invalid json", 400, corsHeaders(corsOrigin));
    }

    // Light validation: only allow the shape we expect from the frontend so
    // callers can't smuggle through arbitrary tool / system prompt overrides.
    if (typeof body !== "object" || body === null) {
      return text("body must be an object", 400, corsHeaders(corsOrigin));
    }
    const safe = sanitizeRequest(body as Record<string, unknown>);
    if (!safe) {
      return text("rejected: unexpected fields", 400, corsHeaders(corsOrigin));
    }

    // Per-IP rate limit. Refusal happens BEFORE we hit Anthropic so abuse is cheap.
    const clientIp = req.headers.get("cf-connecting-ip") ?? "unknown";
    const decision = await checkRateLimit(env, clientIp, ctx);
    if (!decision.allowed) {
      const headers: Record<string, string> = {
        ...corsHeaders(corsOrigin),
        "content-type": "text/plain",
      };
      if (decision.retryAfterSec) {
        headers["retry-after"] = String(decision.retryAfterSec);
      }
      return new Response(decision.reason ?? "rate limit", { status: 429, headers });
    }

    const upstream = await fetch(ANTHROPIC_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
      },
      body: JSON.stringify(safe),
    });

    // Pass through the streaming body unchanged.
    const headers = new Headers(corsHeaders(corsOrigin));
    headers.set("content-type", upstream.headers.get("content-type") ?? "application/json");
    headers.set("cache-control", "no-cache");
    return new Response(upstream.body, { status: upstream.status, headers });
  },
};

// =============================================================================
// Rate limiter
// =============================================================================

async function checkRateLimit(
  env: Env,
  ip: string,
  ctx: ExecutionContext,
): Promise<RateDecision> {
  if (!ip || ip === "unknown") {
    // Bucketed under one shared key. Slightly punitive for unidentified clients,
    // but better than letting them bypass entirely.
    ip = "unknown";
  }
  const now = Math.floor(Date.now() / 1000);
  const key = `rl:${ip}`;

  const stored = await env.RATE_LIMIT_KV.get(key);
  let rec: IPRecord = stored
    ? safeParseRecord(stored)
    : { ts: [], strikes: 0, blockedUntil: 0 };

  // Honour active blocklist.
  if (rec.blockedUntil > now) {
    return {
      allowed: false,
      reason:
        "Your IP is temporarily blocked for repeated rapid-fire queries. " +
        `Try again in ${humanDuration(rec.blockedUntil - now)}.`,
      retryAfterSec: rec.blockedUntil - now,
    };
  }
  if (rec.blockedUntil > 0 && rec.blockedUntil <= now) {
    rec.blockedUntil = 0;
    rec.strikes = 0;
  }

  // Drop entries outside the longest window.
  rec.ts = rec.ts.filter((t) => t >= now - LONGEST_WINDOW_SEC);

  // Check each window.
  for (const lim of LIMITS) {
    const cutoff = now - lim.windowSec;
    const count = rec.ts.reduce((c, t) => (t >= cutoff ? c + 1 : c), 0);
    if (count >= lim.cap) {
      // Per-minute breach -> strike. Two strikes -> 6h block.
      if (lim.windowSec === 60) {
        rec.strikes += 1;
        if (rec.strikes >= STRIKE_THRESHOLD) {
          rec.blockedUntil = now + STRIKE_BLOCK_SEC;
          await persist(env, key, rec, ctx);
          return {
            allowed: false,
            reason:
              "Per-IP block triggered: repeated per-minute rate-limit hits. " +
              "Blocked for 6 hours.",
            retryAfterSec: STRIKE_BLOCK_SEC,
          };
        }
      }
      const oldestInWindow = rec.ts.find((t) => t >= cutoff) ?? now;
      const retryAfter = Math.max(1, oldestInWindow + lim.windowSec - now);
      await persist(env, key, rec, ctx);
      return {
        allowed: false,
        reason:
          `Rate limit reached: ${lim.cap} queries per ${lim.label} per IP. ` +
          `Try again in ${humanDuration(retryAfter)}.`,
        retryAfterSec: retryAfter,
      };
    }
  }

  // Allowed: record the timestamp + reset strikes (we got past per-minute).
  rec.ts.push(now);
  rec.strikes = 0;
  await persist(env, key, rec, ctx);
  return { allowed: true };
}

function safeParseRecord(raw: string): IPRecord {
  try {
    const obj = JSON.parse(raw) as Partial<IPRecord>;
    return {
      ts: Array.isArray(obj.ts) ? (obj.ts as number[]).filter((n) => typeof n === "number") : [],
      strikes: typeof obj.strikes === "number" ? obj.strikes : 0,
      blockedUntil: typeof obj.blockedUntil === "number" ? obj.blockedUntil : 0,
    };
  } catch {
    return { ts: [], strikes: 0, blockedUntil: 0 };
  }
}

async function persist(
  env: Env,
  key: string,
  rec: IPRecord,
  ctx: ExecutionContext,
): Promise<void> {
  // Keep the KV entry alive at least until the longest window expires after the
  // last activity. Adds a small TTL buffer for blocklist persistence.
  const ttlSec = Math.max(LONGEST_WINDOW_SEC, STRIKE_BLOCK_SEC) + 600;
  // Don't block the request on the KV write — fire it in the background.
  ctx.waitUntil(
    env.RATE_LIMIT_KV.put(key, JSON.stringify(rec), { expirationTtl: ttlSec }),
  );
}

function humanDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  return `${Math.ceil(seconds / 3600)}h`;
}

// =============================================================================
// Existing helpers
// =============================================================================

function corsHeaders(origin: string): Record<string, string> {
  return {
    "access-control-allow-origin": origin,
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
    "vary": "origin",
  };
}

function text(body: string, status: number, headers: Record<string, string>): Response {
  return new Response(body, { status, headers: { ...headers, "content-type": "text/plain" } });
}

interface SafeRequest {
  model: string;
  max_tokens: number;
  stream: boolean;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
}

function sanitizeRequest(input: Record<string, unknown>): SafeRequest | null {
  const model = input["model"];
  const maxTokens = input["max_tokens"];
  const stream = input["stream"];
  const messages = input["messages"];

  if (typeof model !== "string") return null;
  if (typeof maxTokens !== "number" || maxTokens > 2048) return null;
  if (typeof stream !== "boolean") return null;
  if (!Array.isArray(messages) || messages.length === 0) return null;

  const safeMessages: SafeRequest["messages"] = [];
  for (const m of messages) {
    if (typeof m !== "object" || m === null) return null;
    const role = (m as Record<string, unknown>)["role"];
    const content = (m as Record<string, unknown>)["content"];
    if (role !== "user" && role !== "assistant") return null;
    if (typeof content !== "string" || content.length > 8000) return null;
    safeMessages.push({ role, content });
  }

  return { model, max_tokens: maxTokens, stream, messages: safeMessages };
}

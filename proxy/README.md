# expression-explain-proxy

Tiny Cloudflare Worker that holds the Anthropic API key for the "Why?" feature.
The frontend (deployed to Hugging Face Spaces) never sees the key.

## Deploy

```bash
cd proxy
npm install
npx wrangler login                     # one-time
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler deploy
```

Note the deployed URL (e.g. `https://expression-explain-proxy.<account>.workers.dev`).
Set `VITE_EXPLAIN_PROXY_URL` to that URL when building the frontend.

## Origin allowlist

Edit `wrangler.toml` and set `ALLOWED_ORIGIN` to a comma-separated list including
your final HF Space URL, e.g.:

```toml
[vars]
ALLOWED_ORIGIN = "https://your-username-expression.hf.space,http://localhost:5173"
```

Then redeploy with `wrangler deploy`.

## Rate limiting

Set this in the **Cloudflare dashboard** (free), not in code:

1. Go to Workers & Pages → your worker → Settings → Triggers / Rate Limiting
2. Add a rule: 30 requests / 5 minutes per IP, action: block 5m

Code-level KV counters cost more and add latency. Use the dashboard rule unless
you need per-user quotas.

## What it does

- Accepts POST `application/json` from allowed origins only (CORS preflight handled).
- Validates the body shape (no system prompt smuggling, no oversized payloads).
- Adds `x-api-key` and `anthropic-version` server-side.
- Streams the SSE response back to the browser unchanged.

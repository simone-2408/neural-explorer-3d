# `src/llm`

Client of the GPT-2 backend (`python/server`, reached through the `/api` proxy of the Vite server) and
loader of the recordings in `public/llm-demo`, used when the backend is not running.

| File | Content |
| --- | --- |
| `client.ts` | `ONLINE_DEMO` (static build without backend, set by the Pages workflow: no status polling, messages link to the local setup), `fetchStatus` (null when the backend is off), `analyze`, `generate`, a generic `post` used by the Experiments tab, demo loading, decoding of the base64 tensors (float32 logits, uint16 attention) |
| `types.ts` | payload types: status, tokens (BPE pieces and bytes), logit-lens layers, analyses, generations, demo entries |

The views do not care where an analysis comes from: a live analysis and a recorded one have the same type
(`LlmAnalysis`, with `source: 'live' | 'demo'`). The Experiments tab's calls live in
`src/experiments/gpt2Engine.ts`.

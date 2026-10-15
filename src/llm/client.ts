/**
 * Client of the local Python backend (python/server, proxied by Vite at /api)
 * and loader of the recorded demo analyses (public/llm-demo), used when the
 * backend is not running. The views do not care where an analysis comes from.
 */
import type { DemoEntry, GenerateResult, LlmAnalysis, LlmAnalysisJson, LlmStatus } from './types';

const API = 'api/llm';

function decodeBase64(data: string): ArrayBuffer {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

/** Decodes the binary fields of an analysis (little-endian, as written by numpy "<f4" / "<u2"). */
export function decodeAnalysis(json: LlmAnalysisJson, source: 'live' | 'demo'): LlmAnalysis {
  const T = json.tokens.length;
  const layers = json.lens.length - 1;
  const u16 = new Uint16Array(decodeBase64(json.attention));
  const attention = Float32Array.from(u16, (v) => v / 65535);
  const heads = Math.round(attention.length / (layers * T * T));
  return { ...json, T, layers, heads, logits: new Float32Array(decodeBase64(json.logits)), attention, source };
}

export class LlmError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** POST to the backend (`api/llm/<path>`); errors carry the server's explanation. */
export async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(new URL(`${API}/${path}`, document.baseURI), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const data = (await response.json()) as { detail?: unknown };
      if (data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not JSON */
    }
    throw new LlmError(detail, response.status);
  }
  return (await response.json()) as T;
}

/** null when the backend is not reachable (not started, or a static build without proxy). */
export async function fetchStatus(timeoutMs = 2500): Promise<LlmStatus | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(new URL(`${API}/status`, document.baseURI), { signal: controller.signal });
    if (!response.ok) return null;
    const data = (await response.json()) as LlmStatus & { offline?: boolean };
    return !data.offline && typeof data.ready === 'boolean' ? data : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function analyze(request: { text?: string; ids?: number[] }, signal?: AbortSignal): Promise<LlmAnalysis> {
  return decodeAnalysis(await post<LlmAnalysisJson>('analyze', request, signal), 'live');
}

export function generate(request: { text: string; temperature: number; topK: number; topP: number; maxNewTokens: number; seed: number }): Promise<GenerateResult> {
  return post<GenerateResult>('generate', request);
}

// ------------------------------------------------------------------- demos

let demoIndex: Promise<DemoEntry[]> | null = null;

export function loadDemoIndex(): Promise<DemoEntry[]> {
  demoIndex ??= fetch(new URL('llm-demo/index.json', document.baseURI))
    .then((r) => (r.ok ? (r.json() as Promise<DemoEntry[]>) : []))
    .catch(() => {
      demoIndex = null;
      return [];
    });
  return demoIndex;
}

export async function loadDemo(entry: DemoEntry): Promise<LlmAnalysis> {
  const response = await fetch(new URL(`llm-demo/${entry.file}`, document.baseURI));
  if (!response.ok) throw new Error(`Demo ${entry.id}: HTTP ${response.status}`);
  return decodeAnalysis((await response.json()) as LlmAnalysisJson, 'demo');
}

/** Attention probability (layer, head, query, key). */
export function attentionAt(a: LlmAnalysis, layer: number, head: number, query: number, key: number): number {
  const T = a.T;
  return a.attention[((layer * a.heads + head) * T + query) * T + key];
}

/** The T×T attention matrix of one head (a view into the analysis). */
export function headMatrix(a: LlmAnalysis, layer: number, head: number): Float32Array {
  const size = a.T * a.T;
  const offset = (layer * a.heads + head) * size;
  return a.attention.subarray(offset, offset + size);
}

/** Readable token text: leading spaces as "·", newlines as "⏎". */
export function showToken(text: string): string {
  return text.replace(/\n/g, '⏎').replace(/^ /, '·').replace(/ /g, '·') || '∅';
}

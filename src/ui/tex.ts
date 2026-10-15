/**
 * Math rendering with KaTeX (Level 3 of the "visual → numbers → math"
 * principle). Rendering is synchronous and fast enough to run on every
 * inspector refresh.
 */
import katex from 'katex';
import 'katex/dist/katex.min.css';

const cache = new Map<string, string>();

export function texToHtml(tex: string, display = false): string {
  const key = `${display ? 'D' : 'I'}|${tex}`;
  let html = cache.get(key);
  if (html === undefined) {
    html = katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', trust: false });
    if (cache.size > 500) cache.clear();
    cache.set(key, html);
  }
  return html;
}

/** Element containing rendered math. */
export function tex(source: string, options: { display?: boolean; class?: string } = {}): HTMLElement {
  const el = document.createElement(options.display ? 'div' : 'span');
  el.className = options.class ?? (options.display ? 'math math--display' : 'math');
  // KaTeX output is generated from our own TeX strings; user text is escaped via texText().
  el.innerHTML = texToHtml(source, options.display);
  return el;
}

/** Escapes arbitrary text for use inside \text{…}. */
export function texText(text: string): string {
  const escaped = text
    .replace(/\\/g, '∖')
    .replace(/\^/g, 'ˆ')
    .replace(/~/g, '∼')
    .replace(/[{}$&#_%]/g, (c) => `\\${c}`);
  return `\\text{${escaped}}`;
}

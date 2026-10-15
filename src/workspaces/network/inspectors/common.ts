/** Building blocks shared by the network inspectors. */
import { ACTIVATIONS } from '../../../math/activations';
import type { NetworkModel } from '../../../neural/NetworkModel';
import {
  biasSymbol,
  neuronTitle,
  outputSymbol,
  preActivationSymbol,
  weightSymbol,
} from '../../../neural/topology';
import { activationPlot, softmaxBars, type PlotPoint } from '../../../ui/charts/activationPlot';
import { h, type Child } from '../../../ui/dom';
import { levelHeading, type Level } from '../../../ui/widgets';
import { texNumber, texParen } from '../../../utils/format';

export function inspectorHeader(o: { kind: string; title: string; subtitle?: Child; actions?: Child[] }): HTMLElement {
  return h(
    'header',
    { class: 'insp-header' },
    h('div', { class: 'insp-header__text' }, h('div', { class: 'insp-kind' }, o.kind), h('h2', { class: 'insp-title' }, o.title), o.subtitle ? h('div', { class: 'insp-subtitle' }, o.subtitle) : null),
    o.actions?.length ? h('div', { class: 'insp-actions' }, o.actions) : null,
  );
}

/** A titled block of the inspector, optionally tagged with an explanation level. */
export function block(level: Level | null, title: Child | null, ...children: Child[]): HTMLElement {
  return h(
    'section',
    { class: `insp-block${level ? ` insp-block--level${level}` : ''}` },
    level ? levelHeading(level, title ?? undefined) : title ? h('h3', { class: 'insp-block__title' }, title) : null,
    children,
  );
}

/** Inline button that selects another element (keeps the inspector navigable like hyperlinks). */
export function entityLink(label: string, onClick: () => void, title?: string): HTMLButtonElement {
  return h('button', { type: 'button', class: 'entity-link', onClick, title: title ?? `Select ${label}` }, label);
}

export function paragraph(...children: Child[]): HTMLParagraphElement {
  return h('p', { class: 'insp-text' }, children);
}

export interface Term {
  /** Source neuron index (in layer - 1). */
  source: number;
  x: number;
  w: number;
  contribution: number;
}

/** Terms w_i · x_i of the weighted sum of neuron (layer, index). */
export function incomingTerms(model: NetworkModel, layer: number, index: number): Term[] {
  const x = model.trace.a[layer - 1];
  const terms: Term[] = [];
  for (let i = 0; i < x.length; i++) {
    const w = model.weight(layer, index, i);
    terms.push({ source: i, x: x[i], w, contribution: w * x[i] });
  }
  return terms;
}

const MAX_EXPANDED_TERMS = 4;

/** TeX of z = Σ W x + b, symbolic → substituted → result. */
export function weightedSumTex(model: NetworkModel, layer: number, index: number): string {
  const { spec, trace } = model;
  const n = spec.layers[layer - 1].size;
  const b = model.bias(layer, index);
  const z = trace.z[layer][index];
  const x = trace.a[layer - 1];
  const symTerm = (i: number) => `${weightSymbol({ layer, target: index, source: i })}\\,${outputSymbol(spec, layer - 1, i)}`;
  const numTerm = (i: number) => `${texParen(model.weight(layer, index, i))} \\cdot ${texParen(x[i])}`;
  let symbolic: string;
  let numeric: string;
  if (n <= MAX_EXPANDED_TERMS) {
    symbolic = Array.from({ length: n }, (_, i) => symTerm(i)).join(' + ');
    numeric = Array.from({ length: n }, (_, i) => numTerm(i)).join(' + ');
  } else {
    const prevSymbol = spec.layers[layer - 1].role === 'input' ? 'x_i' : `a^{(${layer - 1})}_i`;
    symbolic = `\\sum_{i=1}^{${n}} W^{(${layer})}_{${index + 1},i}\\, ${prevSymbol}`;
    numeric = `${numTerm(0)} + ${numTerm(1)} + \\cdots + ${numTerm(n - 1)}`;
  }
  return `\\begin{aligned} ${preActivationSymbol(layer, index)} &= ${symbolic} + ${biasSymbol(layer, index)} \\\\ &= ${numeric} + ${texParen(b)} \\\\ &= ${texNumber(z)} \\end{aligned}`;
}

/** TeX of a = f(z) with the numbers substituted. */
export function activationTex(model: NetworkModel, layer: number, index: number): string {
  const { spec, trace } = model;
  const activation = ACTIVATIONS[spec.layers[layer].activation];
  const z = trace.z[layer][index];
  const a = trace.a[layer][index];
  const lhs = outputSymbol(spec, layer, index);
  const pre = preActivationSymbol(layer, index);
  if (activation.kind === 'elementwise') {
    if (activation.id === 'linear') return `${lhs} = ${pre} = ${texNumber(a)}`;
    return `${lhs} = ${activation.tex}\\!\\left(${pre}\\right) = ${activation.texEval(z, (v) => texNumber(v))} = ${texNumber(a)}`;
  }
  // Softmax couples the whole layer.
  const zs = trace.z[layer];
  const exps = Array.from(zs, (v) => Math.exp(v));
  const sum = exps.reduce((s, v) => s + v, 0);
  const denominator =
    zs.length <= MAX_EXPANDED_TERMS ? Array.from(zs, (v) => `e^{${texNumber(v, 2)}}`).join(' + ') : `\\sum_{k=1}^{${zs.length}} e^{z^{(${layer})}_k}`;
  return `\\begin{aligned} ${lhs} &= \\frac{e^{${pre}}}{\\sum_{k} e^{z^{(${layer})}_{k}}} = \\frac{e^{${texNumber(z, 2)}}}{${denominator}} \\\\ &= \\frac{${texNumber(exps[index])}}{${texNumber(sum)}} = ${texNumber(a)} \\end{aligned}`;
}

/** Plot of the layer's activation function with its neurons as points (or softmax bars). */
export function activationFigure(model: NetworkModel, layer: number, highlight: number | null): HTMLElement {
  const { spec, trace } = model;
  const id = spec.layers[layer].activation;
  const info = ACTIVATIONS[id];
  if (info.kind === 'vector') {
    const labels = Array.from({ length: spec.layers[layer].size }, (_, j) => neuronTitle(spec, layer, j));
    return softmaxBars(trace.z[layer], trace.a[layer], labels, highlight);
  }
  const points: PlotPoint[] = [];
  const size = spec.layers[layer].size;
  const limit = Math.min(size, 256);
  for (let j = 0; j < limit; j++) {
    points.push({ z: trace.z[layer][j], a: trace.a[layer][j], label: neuronTitle(spec, layer, j), highlight: j === highlight });
  }
  if (highlight !== null && highlight >= limit) {
    points.push({ z: trace.z[layer][highlight], a: trace.a[layer][highlight], label: neuronTitle(spec, layer, highlight), highlight: true });
  }
  return activationPlot(id, points);
}

/** Qualitative state of a neuron for its activation (honest, purely mathematical wording). */
export function neuronState(model: NetworkModel, layer: number, index: number): { label: string; detail: string; tone: 'on' | 'off' | 'warn' } | null {
  const id = model.spec.layers[layer].activation;
  const z = model.trace.z[layer][index];
  if (id === 'relu') {
    return z > 0
      ? { label: 'Active', detail: 'z > 0, so ReLU passes z through unchanged.', tone: 'on' }
      : { label: 'Inactive', detail: 'z ≤ 0, so ReLU outputs 0 for this input (and its local slope is 0).', tone: 'off' };
  }
  if (id === 'sigmoid' || id === 'tanh') {
    return Math.abs(z) > 4
      ? { label: 'Saturated', detail: `|z| > 4: the ${id} curve is almost flat here, so small changes of z barely change the output.`, tone: 'warn' }
      : { label: 'Responsive', detail: `|z| ≤ 4: the ${id} curve still has a noticeable slope here.`, tone: 'on' };
  }
  return null;
}

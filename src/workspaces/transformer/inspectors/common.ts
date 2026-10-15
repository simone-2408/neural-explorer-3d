/** Helpers shared by the Transformer inspectors. */
import type { TransformerModel } from '../../../transformer/model';
import { h } from '../../../ui/dom';
import { segmented } from '../../../ui/widgets';
import type { TransformerWorkspace } from '../TransformerWorkspace';

/** Long name of a residual stage. */
export function stageName(stage: number): string {
  if (stage === 0) return 'embedding + position';
  const layer = Math.floor((stage - 1) / 2) + 1;
  return (stage - 1) % 2 === 0 ? `after attention ${layer}` : `after MLP ${layer}`;
}

/** Short name of a residual stage, for compact controls. */
export function stageShort(stage: number): string {
  if (stage === 0) return 'x₀';
  const layer = Math.floor((stage - 1) / 2) + 1;
  return (stage - 1) % 2 === 0 ? `+Attn ${layer}` : `+MLP ${layer}`;
}

export function stageKind(stage: number): { kind: 'embed' | 'attention' | 'mlp'; layer: number } {
  if (stage === 0) return { kind: 'embed', layer: -1 };
  const layer = Math.floor((stage - 1) / 2);
  return { kind: (stage - 1) % 2 === 0 ? 'attention' : 'mlp', layer };
}

/** Segmented control that moves a token selection through the stages. */
export function stagePicker(ws: TransformerWorkspace, position: number, stage: number): HTMLElement {
  const control = segmented<string>({
    options: Array.from({ length: ws.stages }, (_, s) => ({ value: String(s), label: stageShort(s), title: stageName(s) })),
    value: String(stage),
    ariaLabel: 'Stage of the residual stream',
    onChange: (v) => ws.select({ kind: 'token', position, stage: Number(v) }),
  });
  return h('div', { class: 'insp-picker' }, control.el);
}

/** Row of token chips that changes the position of the current selection. */
export function positionPicker(ws: TransformerWorkspace, position: number, onPick: (p: number) => void): HTMLElement {
  const tokens = ws.trace?.tokens ?? [];
  return h(
    'div',
    { class: 'token-chips token-chips--compact', role: 'group', 'aria-label': 'Position' },
    tokens.map((t, p) =>
      h(
        'button',
        { type: 'button', class: `tchip tchip--button${p === position ? ' is-selected' : ''}${t.unknown ? ' is-unknown' : ''}`, onClick: () => onPick(p), title: `position ${p}` },
        h('span', { class: 'tchip__text' }, t.text),
      ),
    ),
  );
}

export function quote(text: string): string {
  return `“${text}”`;
}

/** TeX-safe token text. */
export function texWord(model: TransformerModel, id: number): string {
  const w = model.config.vocab[id].replace(/[{}$&#_%\\^~]/g, '');
  return `\\text{“${w}”}`;
}

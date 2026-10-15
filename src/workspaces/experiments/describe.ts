/**
 * Names used by the Experiments tab. Layers and heads are counted from 1, as
 * everywhere in the app (papers on GPT-2 often count from 0: their head
 * “9.6” is layer 10 · head 7 here). Neurons and token positions are
 * identifiers and keep their index (neuron #0, position 0).
 */
import type { ComponentRef, ExperimentResult, Intervention, TensorId } from '../../experiments/types';
import type { TransformerModel } from '../../transformer/model';

export function componentName(c: ComponentRef): string {
  if (c.kind === 'head') return `Layer ${c.layer + 1} · head ${c.head + 1}`;
  if (c.kind === 'mlp') return `MLP of layer ${c.layer + 1}`;
  return `Neuron #${c.unit} of MLP ${c.layer + 1}`;
}

/** The 0-based “layer.head” notation of interpretability papers. */
export function paperName(layer: number, head: number): string {
  return `${layer}.${head}`;
}

const TENSOR_NAMES: Record<TensorId, string> = {
  embed: 'Token embeddings',
  pos: 'Position embeddings',
  q: 'W_Q',
  k: 'W_K',
  v: 'W_V',
  o: 'W_O',
  mlpIn: 'MLP W_in',
  mlpOut: 'MLP W_out',
};

export function tensorName(tensor: TensorId, layer: number, head: number): string {
  if (tensor === 'embed' || tensor === 'pos') return TENSOR_NAMES[tensor];
  if (tensor === 'mlpIn' || tensor === 'mlpOut') return `${TENSOR_NAMES[tensor]} · layer ${layer + 1}`;
  return `${TENSOR_NAMES[tensor]} · layer ${layer + 1} · head ${head + 1}`;
}

export function scaleText(scale: number): string {
  return scale === 0 ? 'off' : `× ${+scale.toFixed(3)}`;
}

/** One-line description of an intervention. */
export function describeIntervention(i: Intervention, result: ExperimentResult | null, format: (text: string) => string, toy: TransformerModel | null): string {
  const token = (p: number) => {
    const t = result?.tokens[p];
    return t ? `“${format(t.text)}”` : `position ${p}`;
  };
  switch (i.kind) {
    case 'head':
      return `${componentName(i)} ${scaleText(i.scale)}`;
    case 'mlp':
      return `${componentName(i)} ${scaleText(i.scale)}`;
    case 'neuron':
      return `${componentName(i)} ${i.mode === 'set' ? `= ${+i.value.toFixed(3)}` : scaleText(i.value)}`;
    case 'embedding':
      return `${token(i.position)} → “${format(i.token.text)}”${i.mix !== 1 ? ` (mix ${Math.round(i.mix * 100)}%)` : ''}`;
    case 'remove':
      return `Remove ${token(i.position)} (position ${i.position})`;
    case 'weight': {
      const parts = [tensorName(i.tensor, i.layer, i.head)];
      if (i.scale !== 1) parts.push(scaleText(i.scale));
      if (i.cell) {
        const rows = toy ? describeCell(toy, i) : `[${i.cell.row}, ${i.cell.col}]`;
        parts.push(`${rows} = ${+i.cell.value.toFixed(3)}`);
      }
      if (i.scale === 1 && !i.cell) parts.push('(unchanged)');
      return parts.join(' ');
    }
  }
}

function describeCell(toy: TransformerModel, i: Extract<Intervention, { kind: 'weight' }>): string {
  const cell = i.cell!;
  if (i.tensor === 'embed') return `[${toy.config.vocab[cell.row] ?? cell.row}, dim ${cell.col + 1}]`;
  return `[${cell.row}, ${cell.col}]`;
}

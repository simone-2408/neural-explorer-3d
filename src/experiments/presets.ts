/**
 * Ready-made experiments. Toy-model presets are defined here (and verified by
 * experiments.test.ts); GPT-2 presets are recorded with their results by
 * python/server/record_demos.py into public/llm-demo/experiments.json, so they
 * also work without the backend.
 *
 * Descriptions report what the numbers show on this input; they avoid
 * claiming what a component "means" (see the caveats in the inspector).
 */
import type { TransformerModel } from '../transformer/model';
import { newId } from './serialization';
import type { Experiment, ExperimentResult, InterventionSpec, SweepResult, TokenRef } from './types';

export interface ExperimentPreset {
  id: string;
  label: string;
  /** What happens, in two or three sentences. */
  description: string;
  /** A suggestion for the next thing to try. */
  tryNext?: string;
  experiment: Experiment;
}

/** A GPT-2 preset with its recorded result (public/llm-demo/experiments.json). */
export interface RecordedExperiment extends ExperimentPreset {
  result: Omit<ExperimentResult, 'model' | 'source'>;
  sweep: Omit<SweepResult, 'source'>;
}

const IT = "the cat didn't cross the river because it was";

interface ToyPresetDef {
  id: string;
  label: string;
  description: string;
  tryNext?: string;
  text: string;
  target: string | null;
  contrast?: string;
  interventions: (InterventionSpec | { kind: 'embedding'; position: number; word: string; mix: number })[];
}

const TOY: ToyPresetDef[] = [
  {
    id: 'toy-it-head',
    label: 'The head that reads “cat”',
    description:
      '“…because it was”: the model hesitates between hungry (said of the cat) and deep (said of the river). From “was”, layer 2 · head 3 puts 98% of its attention on “cat”. Switched off, p(hungry) falls from 46% to 0.3%.',
    tryNext: 'Bars: which head pulls the other way, towards deep? Switch the first one back on and try that one.',
    text: IT,
    target: 'hungry',
    contrast: 'deep',
    interventions: [{ kind: 'head', layer: 1, head: 2, scale: 0 }],
  },
  {
    id: 'toy-amplify',
    label: 'Turn a head up',
    description: 'The same head with its output doubled instead of removed: p(hungry) rises from 46% to 96%. Scaling is a gentler probe than switching off.',
    tryNext: 'Open the head in the inspector and look at the curve of p(hungry) against the scale.',
    text: IT,
    target: 'hungry',
    contrast: 'deep',
    interventions: [{ kind: 'head', layer: 1, head: 2, scale: 2 }],
  },
  {
    id: 'toy-remove',
    label: 'Remove a word',
    description:
      'Without “river” the place is gone: hungry rises from 46% to 62%. In the residual-change wall the columns before the removed word stay flat: a token can only influence the tokens after it.',
    tryNext: 'Remove “cat” instead (click its chip, then Remove).',
    text: IT,
    target: 'hungry',
    contrast: 'deep',
    interventions: [{ kind: 'remove', position: 6 }],
  },
  {
    id: 'toy-fact',
    label: 'Where is the fact?',
    description: 'With the whole MLP of layer 1 switched off, “paris” falls from 99.5% to under 0.1%.',
    tryNext: 'Open MLP 1 in the inspector: zeroing any single one of its 128 neurons barely changes anything — the fact is spread over many of them.',
    text: 'the capital of france is',
    target: 'paris',
    interventions: [{ kind: 'mlp', layer: 0, scale: 0 }],
  },
  {
    id: 'toy-swap',
    label: 'Swap a country',
    description: 'Inside the model the embedding of “france” is replaced by the embedding of “italy” (the text is unchanged): the answer becomes rome.',
    tryNext: 'Drag the mix slider of the intervention: the answer flips between 0.5 and 0.75.',
    text: 'the capital of france is',
    target: 'paris',
    contrast: 'rome',
    interventions: [{ kind: 'embedding', position: 4, word: 'italy', mix: 1 }],
  },
  {
    id: 'toy-order',
    label: 'Without word order',
    description:
      'Position embeddings are the model’s only source of word order. Scaled to zero, “paris” drops from 99.5% to under 0.1% and the model predicts “a” — perhaps continuing “france is…” as in its training sentence “france is a country” (a plausible reading, not a proof).',
    text: 'the capital of france is',
    target: 'paris',
    interventions: [{ kind: 'weight', tensor: 'pos', layer: 0, head: 0, scale: 0, cell: null }],
  },
  {
    id: 'toy-neuron',
    label: 'Force a neuron',
    description:
      'Neuron 10 of MLP 2 normally sits near −0.16 here; set to 6, far outside its usual range, it makes country names win over “paris”. Its output weights point towards germany, italy, china: this shows what the neuron can write, not what it “means”.',
    tryNext: 'Open the neuron in the inspector to see its activation on each token and its effect on the vocabulary.',
    text: 'the capital of france is',
    target: 'paris',
    interventions: [{ kind: 'neuron', layer: 1, unit: 10, mode: 'set', value: 6 }],
  },
  {
    id: 'toy-weights',
    label: 'Edit a weight matrix',
    description:
      'W_Q of layer 1 · head 2 scaled to 0: every token now asks this head the same question (only the bias is left), so the head attends the same way from every position. The prey becomes uncertain: “cow” falls from 99.7% to 12%, and cat, mouse and capital take most of the rest.',
    tryNext: 'Open the weight edit and change one entry only: single weights rarely matter much.',
    text: 'the lion chased the',
    target: 'cow',
    interventions: [{ kind: 'weight', tensor: 'q', layer: 0, head: 1, scale: 0, cell: null }],
  },
];

/** The toy presets, with words resolved to token ids of `model`. */
export function toyPresets(model: TransformerModel): ExperimentPreset[] {
  const ref = (word: string): TokenRef => {
    const id = model.index.get(word);
    if (id === undefined) throw new Error(`Preset word “${word}” is not in the vocabulary`);
    return { id, text: word };
  };
  return TOY.map((def) => ({
    id: def.id,
    label: def.label,
    description: def.description,
    tryNext: def.tryNext,
    experiment: {
      model: 'toy',
      text: def.text,
      target: def.target ? ref(def.target) : null,
      contrast: def.contrast ? ref(def.contrast) : null,
      interventions: def.interventions.map((i) =>
        i.kind === 'embedding' && 'word' in i
          ? { kind: 'embedding' as const, position: i.position, token: ref(i.word), mix: i.mix, id: newId(), enabled: true }
          : { ...(i as InterventionSpec), id: newId(), enabled: true },
      ),
    },
  }));
}

let recorded: Promise<RecordedExperiment[]> | null = null;

/** GPT-2 presets recorded with their results (empty if the file is missing). */
export function loadRecordedExperiments(): Promise<RecordedExperiment[]> {
  recorded ??= fetch(new URL('llm-demo/experiments.json', document.baseURI))
    .then((r) => (r.ok ? (r.json() as Promise<RecordedExperiment[]>) : []))
    .then((list) => list.map((p) => ({ ...p, experiment: { ...p.experiment, interventions: p.experiment.interventions.map((i) => ({ ...i, id: newId(), enabled: i.enabled !== false })) } })))
    .catch(() => {
      recorded = null;
      return [];
    });
  return recorded;
}

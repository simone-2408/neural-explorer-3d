/**
 * Save / load a training session as JSON: settings (dataset, architecture,
 * optimizer…), the model (format "neural-explorer-3d/mlp", readable from
 * PyTorch), the training history and the optimizer state.
 */
import { deserializeModel, serializeModel, type SerializedModel } from '../../neural/serialization';
import type { HistoryPoint } from '../../training/Trainer';
import { toast } from '../../ui/overlays';
import { downloadText, pickFile, slugify } from '../../utils/download';
import type { TrainingSettings } from '../../training/session';
import type { TrainingWorkspace } from './TrainingWorkspace';

export const SESSION_FORMAT = 'neural-explorer-3d/training-session';

interface SessionFile {
  format: typeof SESSION_FORMAT;
  formatVersion: 1;
  settings: TrainingSettings;
  model: SerializedModel;
  trainer: { stepCount: number; epochCount: number; optimizer: ReturnType<TrainingWorkspace['trainer']['optimizerState']> };
  history: HistoryPoint[];
  meta: { exportedAt: string };
}

export function exportSession(ws: TrainingWorkspace): void {
  const data: SessionFile = {
    format: SESSION_FORMAT,
    formatVersion: 1,
    settings: ws.settings,
    model: serializeModel(ws.model),
    trainer: { stepCount: ws.trainer.stepCount, epochCount: ws.trainer.epochCount, optimizer: ws.trainer.optimizerState() },
    history: ws.trainer.history.slice(-2000),
    meta: { exportedAt: new Date().toISOString() },
  };
  downloadText(`training-${slugify(ws.settings.dataset)}.session.json`, JSON.stringify(data));
  toast('Training session exported (settings, weights, optimizer state, history)', 'success');
}

export async function importSession(ws: TrainingWorkspace): Promise<void> {
  const file = await pickFile('.json,application/json');
  if (!file) return;
  let data: Partial<SessionFile>;
  try {
    data = JSON.parse(await file.text()) as Partial<SessionFile>;
  } catch (error) {
    toast(`Not valid JSON: ${(error as Error).message}`, 'error', 5000);
    return;
  }
  if (data.format !== SESSION_FORMAT || !data.settings || !data.model) {
    toast(`${file.name} is not a training session file`, 'error', 5000);
    return;
  }
  const loaded = deserializeModel(data.model);
  if (!loaded.ok || !loaded.model.weights || !loaded.model.biases) {
    toast(`Cannot load the model: ${loaded.ok ? 'no weights' : loaded.errors[0]}`, 'error', 6000);
    return;
  }
  ws.update(structuredClone(data.settings));
  if (loaded.model.spec.layers.some((l, i) => l.size !== ws.model.spec.layers[i]?.size)) {
    toast('The saved model does not match its settings', 'error', 5000);
    return;
  }
  ws.model.setParameters(loaded.model.weights, loaded.model.biases, 'import');
  ws.trainer.restore({ stepCount: data.trainer?.stepCount ?? 0, epochCount: data.trainer?.epochCount ?? 0, history: data.history ?? [], optimizer: data.trainer?.optimizer });
  ws.noteExternalStep();
  toast(`Loaded ${file.name}`, 'success');
}

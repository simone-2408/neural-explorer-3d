/**
 * What a lesson can drive: the workspaces of the app, reached through the
 * same public actions the panels use. Switching tabs is part of the lesson.
 */
import type { App } from '../app/App';
import type { WorkspaceId } from '../app/state';
import type { Workspace } from '../app/workspace';
import type { CnnWorkspace } from '../workspaces/cnn/CnnWorkspace';
import type { ExperimentsWorkspace } from '../workspaces/experiments/ExperimentsWorkspace';
import type { RnnWorkspace } from '../workspaces/rnn/RnnWorkspace';
import type { LlmWorkspace } from '../workspaces/llm/LlmWorkspace';
import type { NetworkWorkspace } from '../workspaces/network/NetworkWorkspace';
import type { TrainingTab } from '../workspaces/training/TrainingTab';
import type { TrainingWorkspace } from '../workspaces/training/TrainingWorkspace';
import type { TransformerWorkspace } from '../workspaces/transformer/TransformerWorkspace';

/** Resolves when `condition` holds (checked every 100 ms), or after `timeout` ms. */
export function until(condition: () => boolean, timeout = 10000): Promise<boolean> {
  return new Promise((resolve) => {
    const start = performance.now();
    const check = () => {
      if (condition()) resolve(true);
      else if (performance.now() - start > timeout) resolve(false);
      else window.setTimeout(check, 100);
    };
    check();
  });
}

export class LessonContext {
  readonly app: App;
  /** Values remembered by a step for its check (e.g. the input before the learner changed it). */
  readonly memo = new Map<string, unknown>();

  constructor(app: App) {
    this.app = app;
  }

  get store() {
    return this.app.store;
  }

  /** The workspace of a tab, without switching to it (for checks). */
  peek<T extends Workspace>(id: WorkspaceId): T {
    return this.app.getWorkspace(id) as T;
  }

  get activeTab(): WorkspaceId {
    return this.app.store.state.workspace;
  }

  network(): NetworkWorkspace {
    return this.app.switchTo('network') as NetworkWorkspace;
  }

  /** The Training tab in one of its modes (Train, Break it, Compare); returns the single-network workspace. */
  training(mode: 'train' | 'break' | 'compare' = 'train'): TrainingWorkspace {
    const tab = this.app.switchTo('training') as TrainingTab;
    tab.setMode(mode);
    return tab.train;
  }

  /** The Network tab's workspace, without switching (for checks). */
  get networkWs(): NetworkWorkspace {
    return this.peek<NetworkWorkspace>('network');
  }

  /** The Training tab's single-network workspace, without switching (for checks). */
  get trainingWs(): TrainingWorkspace {
    return this.peek<TrainingTab>('training').train;
  }

  get transformerWs(): TransformerWorkspace {
    return this.peek<TransformerWorkspace>('transformer');
  }

  get experimentsWs(): ExperimentsWorkspace {
    return this.peek<ExperimentsWorkspace>('experiments');
  }

  cnn(): CnnWorkspace {
    return this.app.switchTo('cnn') as CnnWorkspace;
  }

  get cnnWs(): CnnWorkspace {
    return this.peek<CnnWorkspace>('cnn');
  }

  rnn(): RnnWorkspace {
    return this.app.switchTo('rnn') as RnnWorkspace;
  }

  get rnnWs(): RnnWorkspace {
    return this.peek<RnnWorkspace>('rnn');
  }

  /** The Transformer tab, once its model is loaded and its scene is built. */
  async transformer(): Promise<TransformerWorkspace> {
    const ws = this.app.switchTo('transformer') as TransformerWorkspace;
    await until(() => !!ws.trace && !!ws.player);
    return ws;
  }

  /** The LLM tab, once an analysis (live or recorded) is shown. */
  async llm(): Promise<LlmWorkspace> {
    const ws = this.app.switchTo('llm') as LlmWorkspace;
    await until(() => !!ws.analysis);
    return ws;
  }

  async experiments(): Promise<ExperimentsWorkspace> {
    const ws = this.app.switchTo('experiments') as ExperimentsWorkspace;
    await until(() => !!ws.toy && !!ws.result);
    return ws;
  }
}

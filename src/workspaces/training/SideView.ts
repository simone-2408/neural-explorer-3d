/**
 * Left half of the Training stage: data + decision boundary, loss landscape
 * or gradient norms (tabs), the loss/accuracy curves, the probe point and the
 * live diagnosis.
 */
import { ACTIVATIONS } from '../../math/activations';
import { neuronTitle } from '../../neural/topology';
import { createGrid, fillGrid, flattenParameters, projectOnPlane, randomPlane, trajectoryPlane } from '../../training/landscape';
import { LineChart } from '../../ui/charts/lineChart';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button, segmented, toggle, type Segmented } from '../../ui/widgets';
import { formatCount, formatValue } from '../../utils/format';
import type { ScenePalette } from '../../visualization/palette';
import { BoundaryView, classColors } from './BoundaryView';
import { LandscapeView } from './LandscapeView';
import { scenarioCard } from './ScenarioPanel';
import type { TrainingWorkspace } from './TrainingWorkspace';

export type SideTab = 'boundary' | 'landscape' | 'gradients';
type Tab = SideTab;

const LAYER_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#D55E00', '#0072B2', '#999999', '#F0E442'];

export class SideView {
  readonly el: HTMLElement;
  private readonly ws: TrainingWorkspace;
  private readonly palette: () => ScenePalette;
  private tab: Tab = 'boundary';
  private readonly tabs: Segmented<Tab>;
  private readonly boundary: BoundaryView;
  private readonly landscape = new LandscapeView();
  private readonly landscapeInfo = h('div', { class: 'side__note' });
  private readonly landscapeStatus = h('div', { class: 'side__status' });
  private readonly gradientBars = h('div', { class: 'grad-bars' });
  private readonly gradientChart = new LineChart({ title: '‖∂L/∂W‖ per layer over training (log scale)', log: true, xLabel: 'epoch', height: 120, format: (v) => formatValue(v, 5) });
  private readonly lossChart = new LineChart({ title: 'Loss', log: true, xLabel: 'epoch', format: (v) => formatValue(v, 4) });
  private readonly accuracyChart = new LineChart({ title: 'Accuracy', yMin: 0, yMax: 1, xLabel: 'epoch', format: (v) => `${formatValue(v * 100, 1)}%` });
  private readonly probeInfo = h('div', { class: 'side__probe' });
  private readonly diagnosis = h('div', { class: 'side__diagnosis', 'aria-live': 'polite' });
  private readonly scenario = h('div', { class: 'side__scenario' });
  private scenarioKey = '';
  private readonly content = h('div', { class: 'side__content' });
  private readonly boundaryPane: HTMLElement;
  private readonly landscapePane: HTMLElement;
  private readonly gradientPane: HTMLElement;
  private followTrajectory = true;
  private lastRefit = 0;

  constructor(ws: TrainingWorkspace, palette: () => ScenePalette) {
    this.ws = ws;
    this.palette = palette;
    this.boundary = new BoundaryView({ model: ws.model, dataset: () => ws.dataset, palette, onProbe: (probe) => ws.setProbe(probe) });
    this.tabs = segmented<Tab>({
      options: [
        { value: 'boundary', label: 'Data & boundary', title: 'The dataset and what the network predicts everywhere in the plane' },
        { value: 'landscape', label: 'Loss landscape', title: 'The loss on a 2-D slice of parameter space' },
        { value: 'gradients', label: 'Gradients', title: 'Size of the gradient reaching each layer' },
      ],
      value: this.tab,
      ariaLabel: 'Side view',
      onChange: (tab) => this.setTab(tab),
    });

    this.boundaryPane = h(
      'div',
      { class: 'side__pane' },
      this.boundary.el,
      h('div', { class: 'boundary__legend' }, h('span', null, '● class 0  ▲ class 1  ■ class 2'), h('span', null, 'filled = training · hollow = test · ring = misclassified · line = decision boundary')),
    );
    this.landscapePane = h(
      'div',
      { class: 'side__pane', hidden: true },
      h(
        'div',
        { class: 'button-row' },
        button({ label: 'Random plane', small: true, title: 'Slice through the current parameters along two random directions', onClick: () => this.computeLandscape('random') }),
        button({ label: 'Fit to trajectory', small: true, title: 'Slice spanned by the two principal components of the training path', onClick: () => this.computeLandscape('pca') }),
        toggle({ label: 'Follow while training', checked: this.followTrajectory, onChange: (v) => (this.followTrajectory = v) }).el,
        toggle({
          label: 'Log scale',
          checked: true,
          onChange: (v) => {
            this.landscape.logScale = v;
            if (ws.grid) this.landscape.setGrid(ws.grid);
            this.updateLandscapePath();
          },
        }).el,
      ),
      this.landscape.el,
      this.landscapeStatus,
      this.landscapeInfo,
    );
    this.gradientPane = h(
      'div',
      { class: 'side__pane', hidden: true },
      h('p', { class: 'side__note' }, 'How large the gradient is when it reaches each weight matrix. If the bars shrink towards the first layers, the error signal fades while travelling back through the network (vanishing gradients) and the early layers learn slowly.'),
      this.gradientBars,
      this.gradientChart.el,
    );
    replaceChildren(this.content, this.boundaryPane, this.landscapePane, this.gradientPane);
    this.el = h(
      'div',
      { class: 'side' },
      h('div', { class: 'side__header' }, this.tabs.el),
      this.scenario,
      this.content,
      this.diagnosis,
      this.probeInfo,
      h('div', { class: 'side__charts' }, this.lossChart.el, this.accuracyChart.el),
    );
  }

  /** Creates the WebGL viewport of the landscape (needs the element in the document). */
  attach(): void {
    this.landscape.attach(this.palette().theme);
  }

  setTheme(theme: 'dark' | 'light'): void {
    this.landscape.setTheme(theme);
    this.update(true);
  }

  dispose(): void {
    this.boundary.dispose();
    this.landscape.dispose();
    this.el.remove();
  }

  /** Redraws the charts; `full` also redraws the boundary map and the landscape. */
  update(full: boolean): void {
    const ws = this.ws;
    const history = ws.trainer.history;
    const palette = this.palette();
    const epochs = history.map((p) => p.epoch);
    const series = (values: number[], label: string, color: string, dash?: number[]) => ({ label, color, values, xs: epochs, dash });
    const hasTest = history.some((p) => p.test);
    this.lossChart.draw([
      series(history.map((p) => p.train.loss), 'train', palette.positive),
      ...(hasTest ? [series(history.map((p) => p.test?.loss ?? NaN), 'test', palette.negative, [5, 4])] : []),
    ]);
    this.accuracyChart.draw([
      series(history.map((p) => p.train.accuracy), 'train', palette.positive),
      ...(hasTest ? [series(history.map((p) => p.test?.accuracy ?? NaN), 'test', palette.negative, [5, 4])] : []),
    ]);

    this.renderProbe();
    this.renderDiagnosis();
    this.renderScenario();
    if (this.tab === 'gradients') this.renderGradients();
    if (this.tab === 'landscape') this.tickLandscape();
    if (full && this.tab === 'boundary') this.boundary.setProbe(ws.probe);
  }

  // -------------------------------------------------------------- tabs

  setTab(tab: Tab): void {
    this.tab = tab;
    this.tabs.set(tab);
    this.boundaryPane.hidden = tab !== 'boundary';
    this.landscapePane.hidden = tab !== 'landscape';
    this.gradientPane.hidden = tab !== 'gradients';
    if (tab === 'landscape' && !this.ws.grid) this.computeLandscape(this.ws.trajectory.length >= 3 ? 'pca' : 'random');
    this.update(true);
  }

  private renderProbe(): void {
    const ws = this.ws;
    const [x, y] = ws.probeInput();
    const label = ws.probeLabel();
    const out = ws.model.output;
    const spec = ws.model.spec;
    const outputLayer = spec.layers.length - 1;
    const prediction =
      out.length === 1
        ? `ŷ = p(class 1) = ${formatValue(out[0], 3)}`
        : `ŷ = (${Array.from(out, (v) => formatValue(v, 2)).join(', ')})`;
    const swatch = (cls: number) => {
      const s = h('span', { class: 'fn-dot' });
      s.style.background = classColors(this.palette())[cls];
      return s;
    };
    replaceChildren(
      this.probeInfo,
      icon('focus', 13),
      h('strong', null, ws.probe.kind === 'sample' ? `Point #${ws.probe.index}` : 'Point'),
      ` (${formatValue(x, 2)}, ${formatValue(y, 2)})`,
      label !== null ? [' · true class ', swatch(label), String(label)] : ' · no label (any point of the plane)',
      ` · ${prediction}`,
      h('span', { class: 'muted' }, ` — shown in the 3D network (${neuronTitle(spec, outputLayer, 0)}…)`),
    );
  }

  private renderScenario(): void {
    const ws = this.ws;
    const key = ws.scenario ? `${ws.scenario.id}:${ws.scenarioFixed}` : '';
    if (key === this.scenarioKey) return;
    this.scenarioKey = key;
    this.scenario.hidden = !ws.scenario;
    replaceChildren(this.scenario, ws.scenario ? scenarioCard(ws, ws.scenario) : null);
  }

  private renderDiagnosis(): void {
    const d = this.ws.diagnosis;
    this.diagnosis.hidden = !d;
    if (!d) return;
    this.diagnosis.className = `side__diagnosis side__diagnosis--${d.severity}`;
    replaceChildren(this.diagnosis, icon(d.severity === 'info' ? 'info' : 'warning', 14), h('div', null, h('strong', null, d.title), h('div', null, d.detail)));
  }

  private renderGradients(): void {
    const ws = this.ws;
    const norms = ws.trainer.gradientNorms();
    const spec = ws.model.spec;
    const max = Math.max(...norms.slice(1), 1e-12);
    replaceChildren(
      this.gradientBars,
      norms.slice(1).map((norm, k) => {
        const l = k + 1;
        const t = norm > 0 ? Math.max(0, (Math.log10(norm / max) + 6) / 6) : 0;
        const fill = h('span', { class: 'grad-bars__fill' });
        fill.style.width = `${(t * 100).toFixed(1)}%`;
        fill.style.background = LAYER_COLORS[k % LAYER_COLORS.length];
        return h(
          'div',
          { class: 'grad-bars__row' },
          h('span', { class: 'grad-bars__label' }, `W${l}`, h('span', { class: 'muted' }, ` ${spec.layers[l - 1].name} → ${spec.layers[l].name} · ${ACTIVATIONS[spec.layers[l].activation].label}`)),
          h('span', { class: 'grad-bars__track' }, fill),
          h('span', { class: 'grad-bars__value' }, norm > 0 ? norm.toExponential(1) : '0'),
        );
      }),
      h('p', { class: 'insp-caption' }, 'Bars use a logarithmic scale over 6 orders of magnitude, relative to the largest layer.'),
    );
    const history = ws.trainer.history;
    this.gradientChart.draw(
      norms.slice(1).map((_, k) => ({
        label: `W${k + 1}`,
        color: LAYER_COLORS[k % LAYER_COLORS.length],
        values: history.map((p) => p.gradNorms[k + 1] ?? NaN),
        xs: history.map((p) => p.epoch),
      })),
    );
  }

  // ---------------------------------------------------------- landscape

  private computeLandscape(kind: 'random' | 'pca'): void {
    const ws = this.ws;
    let plane = kind === 'pca' ? trajectoryPlane(ws.trajectory) : null;
    if (!plane) plane = randomPlane(ws.model, ws.settings.init.seed + ws.trainer.stepCount);
    ws.plane = plane;
    const size = ws.model.topology.connectionCount > 200 ? 25 : 33;
    ws.grid = createGrid(plane, size, 1);
    this.lastRefit = performance.now();
    this.tickLandscape();
  }

  private tickLandscape(): void {
    const ws = this.ws;
    if (!ws.grid || !ws.plane) return;
    // Periodically refit the plane to the trajectory while training.
    if (ws.running && this.followTrajectory && ws.grid.rowsDone >= ws.grid.size && performance.now() - this.lastRefit > 3000 && ws.trajectory.length >= 3) {
      this.computeLandscape('pca');
      return;
    }
    if (ws.grid.rowsDone < ws.grid.size) {
      const done = fillGrid(ws.grid, ws.model.spec, ws.dataset, ws.dataset.train, 10);
      this.landscapeStatus.textContent = done ? '' : `Computing the surface… ${Math.round((ws.grid.rowsDone / ws.grid.size) * 100)}% (${formatCount(ws.grid.size ** 2)} loss evaluations)`;
      if (done) this.landscape.setGrid(ws.grid);
      else requestAnimationFrame(() => this.tab === 'landscape' && this.tickLandscape());
    }
    this.updateLandscapePath();
    const p = ws.plane;
    const params = flattenParameters(ws.model).length;
    replaceChildren(
      this.landscapeInfo,
      h('strong', null, 'This is a 2-D slice, not the whole landscape. '),
      `The loss of this network is a function of ${formatCount(params)} parameters; the surface shows it only on the plane θ = θ꜀ + α·d₁ + β·d₂. `,
      p.kind === 'pca'
        ? `d₁ and d₂ are the two principal components of the training path (they capture ${formatValue((p.explained ?? 0) * 100, 1)}% of its variance), so the path lies close to the plane. `
        : 'd₁ and d₂ are random directions, rescaled neuron by neuron like the weights (Li et al., 2018). The training path generally leaves such a plane. ',
      'The dot is the current parameters projected on the plane; the red line, the path of the optimizer.',
    );
  }

  private updateLandscapePath(): void {
    const ws = this.ws;
    if (!ws.plane || !ws.grid || ws.grid.rowsDone < ws.grid.size) return;
    const plane = ws.plane;
    const path = ws.trajectory.map((p) => projectOnPlane(plane, p));
    this.landscape.setPath(path, projectOnPlane(plane, flattenParameters(ws.model)));
  }
}

/** "Break the Network" scenarios: pick a broken configuration, watch it fail, read why, apply the fix. */
import { DisposeBag } from '../../core/DisposeBag';
import { SCENARIOS, type Scenario } from '../../training/scenarios';
import { h, replaceChildren } from '../../ui/dom';
import { icon } from '../../ui/icons';
import { button, section } from '../../ui/widgets';
import type { Panel } from '../network/context';
import type { TrainingWorkspace } from './TrainingWorkspace';

export class ScenarioPanel implements Panel {
  readonly el: HTMLElement;
  private readonly bag = new DisposeBag();
  private readonly list = h('div', { class: 'scenario-list' });

  constructor(private readonly ws: TrainingWorkspace) {
    this.el = section({
      id: 'training-scenarios',
      title: 'Break the network',
      children: [
        h('p', { class: 'panel-text' }, 'Each scenario sets up a classic mistake. Press Train, watch what goes wrong, then read the explanation and apply the fix.'),
        this.list,
      ],
    });
    this.bag.add(ws.events.on('settings', () => this.render()));
    this.render();
  }

  private render(): void {
    replaceChildren(
      this.list,
      SCENARIOS.map((s) =>
        h(
          'button',
          {
            type: 'button',
            class: `scenario-card${this.ws.scenario?.id === s.id ? ' is-active' : ''}`,
            onClick: () => this.ws.applyScenario(s),
            'aria-pressed': String(this.ws.scenario?.id === s.id),
          },
          h('span', { class: 'scenario-card__title' }, icon('bolt', 13), s.title),
          h('span', { class: 'scenario-card__text' }, s.mistake),
        ),
      ),
    );
  }

  dispose(): void {
    this.bag.dispose();
  }
}

/** Explanation card shown in the side view while a scenario is active. */
export function scenarioCard(ws: TrainingWorkspace, scenario: Scenario): HTMLElement {
  return h(
    'div',
    { class: 'scenario-explain' },
    h('div', { class: 'scenario-explain__title' }, icon('bolt', 14), h('strong', null, scenario.title), ws.scenarioFixed ? h('span', { class: 'scenario-explain__badge' }, 'fixed') : null),
    h('p', null, h('strong', null, 'Look at: '), scenario.watch),
    h('p', null, h('strong', null, 'Why: '), scenario.why),
    h('p', null, h('strong', null, 'Fix: '), scenario.fix),
    h(
      'div',
      { class: 'button-row' },
      ws.scenarioFixed
        ? button({ label: 'Break it again', icon: 'bolt', small: true, onClick: () => ws.applyScenario(scenario) })
        : button({ label: 'Apply the fix', icon: 'check', small: true, variant: 'primary', onClick: () => ws.applyScenarioFix() }),
      button({ label: 'Close', small: true, variant: 'ghost', onClick: () => ws.clearScenario() }),
    ),
  );
}

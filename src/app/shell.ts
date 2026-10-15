/**
 * Application shell: top bar (workspace tabs), side panels, 3D stage (optionally
 * split with a 2D area) and bottom bar. Workspaces fill the slots; the shell
 * knows nothing about neural networks.
 */
import type { Store } from '../core/Store';
import { h, replaceChildren } from '../ui/dom';
import { icon } from '../ui/icons';
import { button } from '../ui/widgets';
import type { AppState, WorkspaceId } from './state';
import { WORKSPACES, type WorkspaceSlots } from './workspace';

export interface Shell {
  slots: WorkspaceSlots;
  /** Container of the WebGL canvas. */
  viewport: HTMLElement;
  setSplit(fraction: number | null): void;
  showFatal(title: string, message: string): void;
}

export interface ShellHandlers {
  onHelp(): void;
  onToggleTheme(): void;
  onSelectWorkspace(id: WorkspaceId): void;
  onOpenLessons(): void;
}

export function buildShell(root: HTMLElement, store: Store<AppState>, handlers: ShellHandlers): Shell {
  const left = h('aside', { class: 'panel panel--left', 'aria-label': 'Controls' });
  const right = h('aside', { class: 'panel panel--right', 'aria-label': 'Inspector' });
  const bottom = h('footer', { class: 'bottombar' });
  const viewport = h('div', { class: 'viewport' });
  const overlay = h('div', { class: 'stage__overlay' });
  const side = h('div', { class: 'stage__side', hidden: true });
  const stage = h('main', { class: 'stage', 'aria-label': 'Workspace view' }, side, viewport, overlay);

  const tabs = WORKSPACES.map((entry) =>
    h(
      'button',
      {
        type: 'button',
        role: 'tab',
        class: 'tab',
        dataset: { workspace: entry.id },
        title: entry.summary,
        onClick: () => (entry.id === 'learn' ? handlers.onOpenLessons() : handlers.onSelectWorkspace(entry.id)),
      },
      h('span', null, entry.label),
    ),
  );
  const syncTabs = () => {
    for (const tab of tabs) {
      const active = tab.dataset.workspace === store.state.workspace;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', String(active));
    }
  };

  const themeButton = button({ icon: store.state.view.theme === 'dark' ? 'sun' : 'moon', title: 'Toggle light / dark theme (T)', variant: 'ghost', onClick: handlers.onToggleTheme });
  const leftToggle = button({
    icon: 'panelLeft',
    title: 'Show / hide the controls panel ([)',
    variant: 'ghost',
    onClick: () => store.set((s) => ({ leftPanelOpen: !s.leftPanelOpen })),
  });
  const rightToggle = button({
    icon: 'panelRight',
    title: 'Show / hide the inspector (])',
    variant: 'ghost',
    onClick: () => store.set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  });
  const helpButton = button({ icon: 'help', label: 'Help', title: 'Help and keyboard shortcuts (?)', variant: 'ghost', onClick: handlers.onHelp });

  const topbar = h(
    'header',
    { class: 'topbar' },
    h(
      'div',
      { class: 'brand' },
      h('span', { class: 'brand__logo', 'aria-hidden': 'true' }, icon('network', 20)),
      h('span', { class: 'brand__name' }, 'Neural Explorer ', h('strong', null, '3D')),
    ),
    h('nav', { class: 'tabs', role: 'tablist', 'aria-label': 'Workspaces' }, tabs),
    h('div', { class: 'topbar__actions' }, leftToggle, rightToggle, themeButton, helpButton),
  );

  const app = h('div', { class: 'app' }, topbar, left, stage, right, bottom);
  replaceChildren(root, app);

  const syncPanels = () => {
    app.dataset.left = store.state.leftPanelOpen ? 'open' : 'closed';
    app.dataset.right = store.state.rightPanelOpen ? 'open' : 'closed';
    leftToggle.setAttribute('aria-pressed', String(store.state.leftPanelOpen));
    rightToggle.setAttribute('aria-pressed', String(store.state.rightPanelOpen));
  };
  syncPanels();
  syncTabs();
  store.subscribe((state, previous) => {
    if (state.leftPanelOpen !== previous.leftPanelOpen || state.rightPanelOpen !== previous.rightPanelOpen) syncPanels();
    if (state.workspace !== previous.workspace) syncTabs();
  });
  store.watch(
    (s) => s.view.theme,
    (theme) => replaceChildren(themeButton, icon(theme === 'dark' ? 'sun' : 'moon')),
  );

  return {
    slots: { left, right, bottom, stage: overlay, side },
    viewport,
    setSplit(fraction) {
      side.hidden = fraction === null;
      stage.classList.toggle('is-split', fraction !== null);
      stage.style.setProperty('--split', fraction === null ? '0%' : `${(fraction * 100).toFixed(2)}%`);
    },
    showFatal(title, message) {
      replaceChildren(stage, h('div', { class: 'fatal' }, icon('warning', 28), h('h2', null, title), h('p', null, message)));
    },
  };
}

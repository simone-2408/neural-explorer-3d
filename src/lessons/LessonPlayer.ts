/**
 * The lesson player: a small card over the 3D view with the current step,
 * its task (checked live against the model) and Back / Next. It lives outside
 * the workspace slots, so it survives the tab switches a lesson makes.
 */
import { h, replaceChildren, type Child } from '../ui/dom';
import { icon } from '../ui/icons';
import { tex } from '../ui/tex';
import { button } from '../ui/widgets';
import type { LessonContext } from './context';
import { saveLessonProgress } from './progress';
import type { Lesson, LessonStep } from './types';

let current: LessonPlayer | null = null;

/** Starts a lesson (closing the one in progress, if any). */
export function playLesson(ctx: LessonContext, lesson: Lesson, step = 0, onCatalog?: () => void): LessonPlayer {
  current?.close();
  current = new LessonPlayer(ctx, lesson, onCatalog);
  void current.goTo(step);
  return current;
}

/** Text with *emphasis* (no HTML is interpreted). */
function rich(text: string): Child[] {
  return text.split('*').map((part, i) => (i % 2 ? h('strong', null, part) : part));
}

export class LessonPlayer {
  readonly el: HTMLElement;
  private readonly ctx: LessonContext;
  private readonly lesson: Lesson;
  private readonly onCatalog?: () => void;
  private readonly body = h('div', { class: 'lesson__body' });
  private readonly footer = h('div', { class: 'lesson__footer' });
  private readonly dots = h('div', { class: 'lesson__dots', 'aria-hidden': 'true' });
  private readonly counter = h('span', { class: 'lesson__counter' });
  private index = 0;
  private timer = 0;
  private entering = 0;
  private readonly done = new Set<number>();
  private collapsed = false;

  constructor(ctx: LessonContext, lesson: Lesson, onCatalog?: () => void) {
    this.ctx = ctx;
    this.lesson = lesson;
    this.onCatalog = onCatalog;
    const collapse = button({ icon: 'minus', title: 'Minimize the lesson', small: true, variant: 'ghost', onClick: () => this.setCollapsed(!this.collapsed) });
    this.el = h(
      'section',
      { class: 'lesson', role: 'region', 'aria-label': `Lesson ${lesson.number}: ${lesson.title}` },
      h(
        'header',
        { class: 'lesson__header' },
        h('span', { class: 'lesson__badge' }, icon('book', 14), `Lesson ${lesson.number}`),
        h('span', { class: 'lesson__title' }, lesson.title),
        this.counter,
        collapse,
        button({ icon: 'close', title: 'Leave the lesson', small: true, variant: 'ghost', onClick: () => this.close() }),
      ),
      this.dots,
      this.body,
      this.footer,
    );
    (document.querySelector('.stage') ?? document.body).append(this.el);
  }

  private setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    this.el.classList.toggle('is-collapsed', collapsed);
  }

  async goTo(index: number): Promise<void> {
    const steps = this.lesson.steps;
    this.index = Math.max(0, Math.min(steps.length - 1, index));
    window.clearInterval(this.timer);
    const step = steps[this.index];
    const ticket = ++this.entering;
    this.renderLoading(step);
    try {
      await step.enter?.(this.ctx);
    } catch (error) {
      console.error('[lesson]', error);
    }
    if (ticket !== this.entering || current !== this) return;
    this.render();
    saveLessonProgress(this.lesson.id, { step: this.index, completed: false });
    if (step.task && !this.done.has(this.index)) {
      this.timer = window.setInterval(() => {
        let ok = false;
        try {
          ok = step.task!.check(this.ctx);
        } catch {
          ok = false;
        }
        if (ok) {
          this.done.add(this.index);
          window.clearInterval(this.timer);
          this.render();
        }
      }, 400);
    }
  }

  private renderLoading(step: LessonStep): void {
    replaceChildren(this.body, h('h3', { class: 'lesson__step' }, step.title), h('div', { class: 'muted ex-pending' }, 'Preparing the scene…'));
  }

  private render(): void {
    const steps = this.lesson.steps;
    const step = steps[this.index];
    const last = this.index === steps.length - 1;
    this.counter.textContent = `${this.index + 1} / ${steps.length}`;
    replaceChildren(
      this.dots,
      steps.map((_, k) => h('button', { type: 'button', class: `lesson__dot${k === this.index ? ' is-current' : ''}${k < this.index || this.done.has(k) ? ' is-done' : ''}`, title: steps[k].title, 'aria-label': `Step ${k + 1}: ${steps[k].title}`, onClick: () => void this.goTo(k) })),
    );
    const text = typeof step.text === 'function' ? step.text(this.ctx) : step.text;
    const task = step.task;
    const completed = this.done.has(this.index);
    replaceChildren(
      this.body,
      h('h3', { class: 'lesson__step' }, step.title),
      text.map((p) => h('p', null, rich(p))),
      step.tex ? tex(step.tex, { display: true, class: 'lesson__tex' }) : null,
      task
        ? h(
            'div',
            { class: `lesson__task${completed ? ' is-done' : ''}`, role: 'status' },
            h('span', { class: 'lesson__check', 'aria-hidden': 'true' }, completed ? icon('check', 14) : null),
            h('span', null, h('strong', null, completed ? 'Done. ' : 'Your turn: '), completed ? (task.done ?? rich(task.prompt)) : rich(task.prompt)),
          )
        : null,
    );
    replaceChildren(
      this.footer,
      button({ icon: 'stepBack', label: 'Back', small: true, variant: 'ghost', disabled: this.index === 0, onClick: () => void this.goTo(this.index - 1) }),
      this.onCatalog ? button({ icon: 'book', label: 'Lessons', small: true, variant: 'ghost', onClick: () => (this.close(), this.onCatalog?.()) }) : null,
      last
        ? button({ icon: 'check', label: 'Finish', small: true, variant: 'primary', onClick: () => this.finish() })
        : button({ icon: 'stepForward', label: task && !completed ? 'Skip' : 'Next', small: true, variant: task && !completed ? 'default' : 'primary', class: task && completed ? 'is-ready' : '', onClick: () => void this.goTo(this.index + 1) }),
    );
  }

  private finish(): void {
    saveLessonProgress(this.lesson.id, { step: this.index, completed: true });
    this.close();
    this.onCatalog?.();
  }

  close(): void {
    window.clearInterval(this.timer);
    this.entering++;
    this.el.remove();
    if (current === this) current = null;
  }
}

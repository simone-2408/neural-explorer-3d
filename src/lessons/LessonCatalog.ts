/** The Learn tab: the catalog of lessons (a dialog); starting one hands over to the lesson player. */
import type { App } from '../app/App';
import { WORKSPACES } from '../app/workspace';
import { h } from '../ui/dom';
import { icon } from '../ui/icons';
import { openModal } from '../ui/overlays';
import { LESSON_GROUPS } from './catalog';
import { LessonContext } from './context';
import { playLesson } from './LessonPlayer';
import { lessonProgress } from './progress';
import type { Lesson } from './types';

export function openLessonCatalog(app: App): void {
  const ctx = new LessonContext(app);
  const reopen = () => openLessonCatalog(app);
  let modal: { close(): void } | null = null;
  const start = (lesson: Lesson, step = 0) => {
    modal?.close();
    playLesson(ctx, lesson, step, reopen);
  };
  const card = (lesson: Lesson) => {
    const progress = lessonProgress(lesson.id);
    const tab = WORKSPACES.find((w) => w.id === lesson.tab)?.label ?? lesson.tab;
    const resume = progress && !progress.completed && progress.step > 0;
    return h(
      'button',
      { type: 'button', class: `lesson-card${progress?.completed ? ' is-done' : ''}`, onClick: () => start(lesson, resume ? progress.step : 0) },
      h('span', { class: 'lesson-card__number' }, progress?.completed ? icon('check', 16) : String(lesson.number)),
      h(
        'span',
        { class: 'lesson-card__text' },
        h('span', { class: 'lesson-card__title' }, lesson.title),
        h('span', { class: 'lesson-card__summary' }, lesson.summary),
        h('span', { class: 'lesson-card__meta' }, `${lesson.steps.length} steps · ${tab} tab${resume ? ` · resume at step ${progress.step + 1}` : progress?.completed ? ' · done' : ''}`),
      ),
    );
  };
  modal = openModal({
    title: 'Learn',
    wide: true,
    content: h(
      'div',
      { class: 'lessons' },
      h('p', { class: 'muted' }, 'Each lesson drives the live scenes: it switches tabs, loads examples and selects what it talks about, and asks you to try things yourself. A small card over the 3D view guides you; you can leave at any time and keep exploring.'),
      LESSON_GROUPS.map((group) => h('section', null, h('h3', { class: 'lessons__group' }, group.title), h('div', { class: 'lessons__grid' }, group.lessons.map(card)))),
    ),
  });
}

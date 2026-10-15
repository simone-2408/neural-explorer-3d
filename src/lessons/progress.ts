/** Lesson progress, remembered per browser (best effort: storage may be unavailable). */

const KEY = 'ne3d.lessons';

export interface LessonProgress {
  /** Highest step reached (0-based). */
  step: number;
  completed: boolean;
}

function read(): Record<string, LessonProgress> {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? '{}') as unknown;
    return value && typeof value === 'object' ? (value as Record<string, LessonProgress>) : {};
  } catch {
    return {};
  }
}

export function lessonProgress(id: string): LessonProgress | null {
  return read()[id] ?? null;
}

export function saveLessonProgress(id: string, progress: LessonProgress): void {
  try {
    const all = read();
    const before = all[id];
    all[id] = { step: Math.max(progress.step, before?.step ?? 0), completed: progress.completed || !!before?.completed };
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    /* ignore */
  }
}

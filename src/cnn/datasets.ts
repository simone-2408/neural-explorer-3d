/**
 * Small synthetic image datasets for the CNN tab: 16×16 grayscale images
 * (0 = background, 1 = ink) generated from a seed, so no data has to be
 * downloaded and every run is reproducible.
 *
 *   lines   — a horizontal, vertical or diagonal stroke anywhere in the image
 *   shapes  — square, plus, cross or ring of random size and position
 *   digits  — 0–9 from a 5×7 pixel font, scaled ×2, shifted and noisy
 *
 * The objects appear at random positions on purpose: a convolutional layer
 * applies the same filter everywhere, so it does not need to learn each
 * position separately.
 */
import { createRng, type Rng } from '../math/random';

export const IMAGE_SIZE = 16;

export type ImageDatasetId = 'lines' | 'shapes' | 'digits';

export interface ImageDatasetInfo {
  id: ImageDatasetId;
  label: string;
  description: string;
  classes: string[];
}

export const IMAGE_DATASETS: Record<ImageDatasetId, ImageDatasetInfo> = {
  lines: {
    id: 'lines',
    label: 'Lines',
    description: 'One stroke per image: horizontal, vertical or diagonal, at any position. Good first filters are edge detectors.',
    classes: ['horizontal', 'vertical', 'diagonal ↘', 'diagonal ↗'],
  },
  shapes: {
    id: 'shapes',
    label: 'Shapes',
    description: 'A square, a plus, a cross or a ring, of random size and position.',
    classes: ['square', 'plus', 'cross', 'ring'],
  },
  digits: {
    id: 'digits',
    label: 'Digits',
    description: 'The digits 0–9 from a 5×7 pixel font, scaled ×2, shifted and noisy — a miniature MNIST.',
    classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
  },
};

export interface ImageSample {
  pixels: Float64Array;
  label: number;
}

export interface ImageDataset {
  id: ImageDatasetId;
  info: ImageDatasetInfo;
  train: ImageSample[];
  test: ImageSample[];
  seed: number;
}

/** 5×7 font for the digits (# = ink). */
const FONT: string[][] = [
  ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
  ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
  ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
  ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
  ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
  ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
  ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
  ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
  ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
];

class Canvas {
  readonly pixels = new Float64Array(IMAGE_SIZE * IMAGE_SIZE);

  plot(x: number, y: number, value = 1): void {
    const xi = Math.round(x);
    const yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= IMAGE_SIZE || yi >= IMAGE_SIZE) return;
    const k = yi * IMAGE_SIZE + xi;
    this.pixels[k] = Math.max(this.pixels[k], value);
  }

  /** Straight stroke from (x0, y0) to (x1, y1), one pixel wide. */
  line(x0: number, y0: number, x1: number, y1: number, value = 1): void {
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let s = 0; s <= steps; s++) this.plot(x0 + ((x1 - x0) * s) / steps, y0 + ((y1 - y0) * s) / steps, value);
  }
}

const randInt = (rng: Rng, min: number, max: number) => min + Math.floor(rng.next() * (max - min + 1));

/** Random intensity of the ink, pixel noise and a few missing pixels: no two samples are identical. */
function finish(canvas: Canvas, rng: Rng): Float64Array {
  const p = canvas.pixels;
  const ink = 0.75 + 0.25 * rng.next();
  for (let k = 0; k < p.length; k++) {
    let v = p[k] * ink;
    if (v > 0 && rng.next() < 0.05) v *= 0.3;
    v += rng.normal(0, 0.06);
    p[k] = Math.min(1, Math.max(0, v));
  }
  return p;
}

function drawLine(label: number, rng: Rng): Float64Array {
  const c = new Canvas();
  const length = randInt(rng, 7, 12);
  const thick = rng.next() < 0.35;
  const S = IMAGE_SIZE;
  if (label === 0 || label === 1) {
    const along = randInt(rng, 0, S - length);
    const across = randInt(rng, 1, S - 2);
    for (let t = 0; t < length; t++) {
      if (label === 0) {
        c.plot(along + t, across);
        if (thick) c.plot(along + t, across + 1);
      } else {
        c.plot(across, along + t);
        if (thick) c.plot(across + 1, along + t);
      }
    }
  } else {
    const x0 = randInt(rng, 0, S - length);
    const y0 = randInt(rng, 0, S - length);
    for (let t = 0; t < length; t++) {
      const x = x0 + t;
      const y = label === 2 ? y0 + t : y0 + length - 1 - t;
      c.plot(x, y);
      if (thick) c.plot(x + 1, y);
    }
  }
  return finish(c, rng);
}

function drawShape(label: number, rng: Rng): Float64Array {
  const c = new Canvas();
  const S = IMAGE_SIZE;
  const size = randInt(rng, 6, 10);
  const x0 = randInt(rng, 0, S - size);
  const y0 = randInt(rng, 0, S - size);
  const x1 = x0 + size - 1;
  const y1 = y0 + size - 1;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  switch (label) {
    case 0: // square outline
      c.line(x0, y0, x1, y0);
      c.line(x0, y1, x1, y1);
      c.line(x0, y0, x0, y1);
      c.line(x1, y0, x1, y1);
      break;
    case 1: // plus
      c.line(x0, Math.round(cy), x1, Math.round(cy));
      c.line(Math.round(cx), y0, Math.round(cx), y1);
      break;
    case 2: // cross
      c.line(x0, y0, x1, y1);
      c.line(x0, y1, x1, y0);
      break;
    default: {
      // ring
      const r = (size - 1) / 2;
      for (let a = 0; a < 64; a++) c.plot(cx + r * Math.cos((a / 64) * 2 * Math.PI), cy + r * Math.sin((a / 64) * 2 * Math.PI));
    }
  }
  return finish(c, rng);
}

function drawDigit(label: number, rng: Rng): Float64Array {
  const c = new Canvas();
  const glyph = FONT[label];
  const ox = randInt(rng, 0, IMAGE_SIZE - 10);
  const oy = randInt(rng, 0, IMAGE_SIZE - 14);
  // a slight slant: each row shifted by up to one pixel
  const slant = rng.next() < 0.5 ? 0 : rng.next() < 0.5 ? -1 : 1;
  for (let r = 0; r < 7; r++) {
    const shift = Math.round((slant * (3 - r)) / 3);
    for (let col = 0; col < 5; col++) {
      if (glyph[r][col] !== '#') continue;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) c.plot(ox + 2 * col + dx + shift, oy + 2 * r + dy);
    }
  }
  return finish(c, rng);
}

const DRAW: Record<ImageDatasetId, (label: number, rng: Rng) => Float64Array> = {
  lines: drawLine,
  shapes: drawShape,
  digits: drawDigit,
};

/** One image of class `label` (the same seed always gives the same image). */
export function drawSample(id: ImageDatasetId, label: number, seed: number): Float64Array {
  return DRAW[id](label, createRng(seed));
}

/** A balanced dataset: classes cycle, so every class has the same number of samples. */
export function makeImageDataset(id: ImageDatasetId, seed = 1, trainSize = 600, testSize = 200): ImageDataset {
  const info = IMAGE_DATASETS[id];
  const rng = createRng(seed);
  const make = (count: number): ImageSample[] =>
    Array.from({ length: count }, (_, k) => {
      const label = k % info.classes.length;
      return { pixels: DRAW[id](label, rng), label };
    });
  return { id, info, train: make(trainSize), test: make(testSize), seed };
}

/** Minimal stroke icons (16×16, currentColor). */
const PATHS = {
  reset: '<path d="M3 8a5 5 0 1 0 1.5-3.6"/><path d="M3 2.5V5h2.5"/>',
  focus: '<circle cx="8" cy="8" r="2"/><path d="M2 5V2h3M11 2h3v3M14 11v3h-3M5 14H2v-3"/>',
  face: '<rect x="4" y="3" width="8" height="10" rx="1.5"/><circle cx="8" cy="6" r=".6"/><circle cx="8" cy="10" r=".6"/>',
  rotate: '<path d="M13.5 8A5.5 5.5 0 0 1 3.2 10.7"/><path d="M2.5 8a5.5 5.5 0 0 1 10.3-2.7"/><path d="M13 2.5v3h-3M3 13.5v-3h3"/>',
  sun: '<circle cx="8" cy="8" r="3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1"/>',
  moon: '<path d="M13 9.5A5.5 5.5 0 1 1 6.5 3a4.5 4.5 0 0 0 6.5 6.5z"/>',
  help: '<circle cx="8" cy="8" r="6"/><path d="M6.3 6.2a1.8 1.8 0 1 1 2.5 1.7c-.5.2-.8.6-.8 1.1v.4"/><circle cx="8" cy="11.6" r=".4"/>',
  download: '<path d="M8 2v8M4.5 7 8 10.5 11.5 7M3 13.5h10"/>',
  upload: '<path d="M8 11V3M4.5 6 8 2.5 11.5 6M3 13.5h10"/>',
  copy: '<rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 10.5V4a1 1 0 0 1 1-1h6.5"/>',
  dice: '<rect x="2.5" y="2.5" width="11" height="11" rx="2"/><circle cx="5.5" cy="5.5" r=".7"/><circle cx="10.5" cy="10.5" r=".7"/><circle cx="8" cy="8" r=".7"/>',
  plus: '<path d="M8 3v10M3 8h10"/>',
  minus: '<path d="M3 8h10"/>',
  close: '<path d="M4 4l8 8M12 4l-8 8"/>',
  panelLeft: '<rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M6 3v10"/>',
  panelRight: '<rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M10 3v10"/>',
  arrowRight: '<path d="M3 8h10M9.5 4.5 13 8l-3.5 3.5"/>',
  check: '<path d="M3 8.5 6.5 12 13 4.5"/>',
  warning: '<path d="M8 2.5 14 13H2z"/><path d="M8 6.5v3"/><circle cx="8" cy="11.2" r=".4"/>',
  info: '<circle cx="8" cy="8" r="6"/><path d="M8 7.2V11"/><circle cx="8" cy="5" r=".4"/>',
  refresh: '<path d="M13 3.5V6.5H10"/><path d="M12.6 6.4A5 5 0 1 0 13 10"/>',
  play: '<path d="M5 3.2v9.6L12.5 8z" fill="currentColor"/>',
  pause: '<path d="M5.2 3.5v9M10.8 3.5v9" stroke-width="2.2"/>',
  stepForward: '<path d="M4 3.5v9L10 8z" fill="currentColor"/><path d="M12 3.5v9" stroke-width="1.8"/>',
  stepBack: '<path d="M12 3.5v9L6 8z" fill="currentColor"/><path d="M4 3.5v9" stroke-width="1.8"/>',
  restart: '<path d="M3.5 3.5v9" stroke-width="1.8"/><path d="M13 3.5v9L6.5 8z" fill="currentColor"/>',
  signal: '<circle cx="3" cy="8" r="1.6"/><circle cx="13" cy="4" r="1.6"/><circle cx="13" cy="12" r="1.6"/><path d="M4.5 7.3 11.4 4.6M4.5 8.7l6.9 2.7"/><path d="M7.5 5.6l1.6 .2-.9 1.3" />',
  fx: '<path d="M2.5 13.5c2.5 0 2.2-11 5-11"/><path d="M3.5 8h4"/><path d="M9.5 6.5l4 6M13.5 6.5l-4 6"/>',
  camera: '<rect x="2" y="4.5" width="9" height="7" rx="1.2"/><path d="M11 7l3-1.8v5.6L11 9"/>',
  dataset: '<circle cx="4" cy="5" r="1.2"/><circle cx="7" cy="10" r="1.2"/><circle cx="11" cy="6" r="1.2"/><circle cx="12" cy="11" r="1.2"/><path d="M2 14h12M2 2v12"/>',
  gauge: '<path d="M2.5 11a5.5 5.5 0 1 1 11 0"/><path d="M8 11l2.5-3.5"/>',
  layers: '<path d="M8 2.5 14 5.5 8 8.5 2 5.5z"/><path d="M2 8.2l6 3 6-3"/><path d="M2 10.8l6 3 6-3"/>',
  flask: '<path d="M6 2.5h4M6.8 2.5v4L3 13a.8.8 0 0 0 .7 1.2h8.6A.8.8 0 0 0 13 13L9.2 6.5v-4"/><path d="M4.6 10h6.8"/>',
  book: '<path d="M8 4.5C6.5 3.3 4.5 3 2.5 3.2v9.3c2-.2 4 .1 5.5 1.3 1.5-1.2 3.5-1.5 5.5-1.3V3.2C11.5 3 9.5 3.3 8 4.5z"/><path d="M8 4.5v9.3"/>',
  sparkles: '<path d="M6 2.5 7.2 6 10.5 7.2 7.2 8.4 6 12 4.8 8.4 1.5 7.2 4.8 6z"/><path d="M12 2v3M10.5 3.5h3M12.5 10v3M11 11.5h3"/>',
  server: '<rect x="2.5" y="2.5" width="11" height="4.5" rx="1"/><rect x="2.5" y="9" width="11" height="4.5" rx="1"/><circle cx="5" cy="4.75" r=".5"/><circle cx="5" cy="11.25" r=".5"/>',
  scissors: '<circle cx="4" cy="4.5" r="1.8"/><circle cx="4" cy="11.5" r="1.8"/><path d="M5.6 5.6 13.5 12M5.6 10.4 13.5 4"/>',
  chart: '<path d="M2.5 13.5h11"/><path d="M3.5 11 6.5 7.5 9 9.5 13 4"/>',
  mountain: '<path d="M1.5 13 6 6l2.5 3.5L10.5 7l4 6z"/><circle cx="10.5" cy="4" r="1"/>',
  compare: '<rect x="2" y="3" width="5" height="10" rx="1"/><rect x="9" y="3" width="5" height="10" rx="1"/>',
  bolt: '<path d="M9 1.5 3.5 9H8l-1 5.5L12.5 7H8z"/>',
  text: '<path d="M3 4V3h10v1M8 3v10M6 13h4"/>',
  network:
    '<circle cx="3" cy="5" r="1.3"/><circle cx="3" cy="11" r="1.3"/><circle cx="8" cy="3.5" r="1.3"/><circle cx="8" cy="8" r="1.3"/><circle cx="8" cy="12.5" r="1.3"/><circle cx="13" cy="8" r="1.3"/><path d="M4.2 5.2 6.8 4M4.2 5.4 6.8 7.7M4.2 10.6 6.8 8.3M4.2 10.8 6.8 12.2M9.2 4.1 11.8 7.4M9.3 8h2.4M9.2 11.9 11.8 8.6"/>',
} as const;

export type IconName = keyof typeof PATHS;

export function icon(name: IconName, size = 16): SVGSVGElement {
  const template = document.createElement('template');
  template.innerHTML = `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name]}</svg>`;
  return template.content.firstElementChild as SVGSVGElement;
}

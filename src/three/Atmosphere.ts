/**
 * Atmosphere — the visual-effects layer of the main viewport, shared by every
 * tab: a deep backdrop with faint nebulae, drifting dust, a glowing floor grid
 * that fades into the distance, and post-processing (HDR rendering with 4×
 * MSAA, bloom on the light emitted by large values, a light vignette and
 * colour fringe, Khronos PBR Neutral tone mapping that keeps the palette's
 * colours as they are).
 *
 * The data keep their encodings: the backdrop, the dust and the floor carry
 * no values, and only elements whose value is large emit the light that the
 * bloom spreads (see fx.ts). In the light theme the effects are toned down to
 * a gradient and a floor; without effects the viewport renders as before.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import type { ThemeId } from '../visualization/palette';
import { FX } from './fx';

export interface AtmosphereColors {
  theme: ThemeId;
  background: string;
  grid: string;
}

const BACKDROP_VERTEX = /* glsl */ `
  varying vec3 vDirection;
  void main() {
    vDirection = position;
    vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = clip.xyww; // on the far plane, behind everything
  }
`;

const BACKDROP_FRAGMENT = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uTintA;
  uniform vec3 uTintB;
  uniform vec2 uGradient; // brightness below / above the horizon
  uniform float uNebula;
  uniform float uTime;
  varying vec3 vDirection;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }
  float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
      mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
      f.z);
  }
  float fbm(vec3 p) {
    float sum = 0.0;
    float amplitude = 0.5;
    for (int k = 0; k < 5; k++) {
      sum += amplitude * noise(p);
      p *= 2.03;
      amplitude *= 0.5;
    }
    return sum;
  }

  void main() {
    vec3 d = normalize(vDirection);
    float h = d.y;
    vec3 color = uBase * mix(uGradient.x, uGradient.y, smoothstep(-0.65, 0.75, h));
    if (uNebula > 0.0) {
      vec3 q = d * 2.3 + vec3(uTime * 0.006, 0.0, -uTime * 0.004);
      float a = smoothstep(0.48, 0.86, fbm(q));
      float b = smoothstep(0.52, 0.9, fbm(q * 1.6 + 7.31));
      float wisps = smoothstep(0.55, 0.8, fbm(q * 4.1 - 3.7));
      color += uNebula * (uTintA * a * 0.075 + uTintB * b * 0.065 + vec3(0.45, 0.55, 0.85) * wisps * a * 0.03);
      // a cold glow along the horizon
      color += uNebula * vec3(0.18, 0.26, 0.42) * 0.09 * exp(-pow(h * 3.5, 2.0));
    }
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;

const DUST_VERTEX = /* glsl */ `
  attribute float aSeed;
  attribute float aSize;
  attribute vec3 aColor;
  uniform float uTime;
  uniform float uMotion;
  uniform float uScale;
  varying vec3 vColor;
  varying float vTwinkle;

  void main() {
    float t = uTime * uMotion;
    vec3 p = position + 0.05 * vec3(
      sin(t * 0.07 + aSeed * 31.0),
      sin(t * 0.05 + aSeed * 17.0) * 0.6,
      cos(t * 0.06 + aSeed * 23.0));
    vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
    gl_PointSize = clamp(aSize * uScale / max(-mvPosition.z, 0.1), 0.0, 9.0);
    gl_Position = projectionMatrix * mvPosition;
    vTwinkle = 0.55 + 0.45 * sin(t * (0.7 + aSeed * 1.9) + aSeed * 57.0);
    vColor = aColor;
  }
`;

const DUST_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vTwinkle;
  void main() {
    float r = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, r);
    a *= a;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor * vTwinkle * a, 1.0);
    #include <colorspace_fragment>
  }
`;

const FLOOR_VERTEX = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const FLOOR_FRAGMENT = /* glsl */ `
  uniform vec3 uLine;
  uniform vec3 uSpot;
  uniform vec2 uCenter;
  uniform float uRadius;
  uniform float uSpacing;
  uniform float uOpacity;
  uniform float uSpotStrength;
  varying vec3 vWorld;

  float gridLine(vec2 p) {
    vec2 g = abs(fract(p - 0.5) - 0.5) / max(fwidth(p), vec2(1e-4));
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    vec2 p = vWorld.xz;
    float d = length(p - uCenter);
    float minor = gridLine(p / uSpacing);
    float major = gridLine(p / (uSpacing * 5.0));
    float fade = 1.0 - smoothstep(uRadius * 0.45, uRadius * 2.6, d);
    float lines = max(minor * 0.42, major) * fade;
    float spot = exp(-pow(d / (uRadius * 1.1), 2.0)) * uSpotStrength;
    vec3 color = uLine * lines + uSpot * spot;
    float alpha = clamp(lines * uOpacity + spot, 0.0, 1.0);
    if (alpha < 0.003) discard;
    gl_FragColor = vec4(color / max(alpha, 1e-3), alpha);
    #include <colorspace_fragment>
  }
`;

/** Vignette and a faint colour fringe towards the edges of the frame. */
const FinishShader = {
  name: 'FinishShader',
  uniforms: {
    tDiffuse: { value: null },
    uVignette: { value: 0.38 },
    uAberration: { value: 0.004 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uVignette;
    uniform float uAberration;
    varying vec2 vUv;
    void main() {
      vec2 c = vUv - 0.5;
      float r = length(c);
      vec2 shift = c * r * uAberration;
      vec4 base = texture2D(tDiffuse, vUv);
      vec3 color = vec3(texture2D(tDiffuse, vUv + shift).r, base.g, texture2D(tDiffuse, vUv - shift).b);
      color *= 1.0 - uVignette * smoothstep(0.3, 0.9, r);
      gl_FragColor = vec4(color, base.a);
    }
  `,
};

const DUST_COUNT = 1500;

export class Atmosphere {
  /** Backdrop, dust and floor: added to the scene by the viewport. */
  readonly group = new THREE.Group();
  readonly composer: EffectComposer;

  private readonly backdrop: THREE.Mesh<THREE.SphereGeometry, THREE.ShaderMaterial>;
  private readonly dust: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  private readonly floor: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private readonly bloom: UnrealBloomPass;
  private readonly finish: ShaderPass;
  private readonly target: THREE.WebGLRenderTarget;
  private readonly center = new THREE.Vector3();
  private radius = 6;
  private theme: ThemeId = 'dark';
  private colors: AtmosphereColors | null = null;

  constructor(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.backdrop = new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 24),
      new THREE.ShaderMaterial({
        name: 'BackdropMaterial',
        vertexShader: BACKDROP_VERTEX,
        fragmentShader: BACKDROP_FRAGMENT,
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: {
          uBase: { value: new THREE.Color() },
          uTintA: { value: new THREE.Color() },
          uTintB: { value: new THREE.Color() },
          uGradient: { value: new THREE.Vector2(0.55, 1.5) },
          uNebula: { value: 1 },
          uTime: FX.uTime,
        },
      }),
    );
    this.backdrop.renderOrder = -1000;
    this.backdrop.frustumCulled = false;

    this.dust = new THREE.Points(
      this.createDustGeometry(),
      new THREE.ShaderMaterial({
        name: 'DustMaterial',
        vertexShader: DUST_VERTEX,
        fragmentShader: DUST_FRAGMENT,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: FX.uTime,
          uMotion: FX.uMotion,
          uScale: { value: 30 },
        },
      }),
    );
    this.dust.frustumCulled = false;

    this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.ShaderMaterial({
        name: 'FloorMaterial',
        vertexShader: FLOOR_VERTEX,
        fragmentShader: FLOOR_FRAGMENT,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        uniforms: {
          uLine: { value: new THREE.Color() },
          uSpot: { value: new THREE.Color() },
          uCenter: { value: new THREE.Vector2() },
          uRadius: { value: 6 },
          uSpacing: { value: 1 },
          uOpacity: { value: 0.8 },
          uSpotStrength: { value: 0.2 },
        },
      }),
    );
    this.floor.renderOrder = -10;
    this.floor.frustumCulled = false;
    this.group.add(this.backdrop, this.dust, this.floor);

    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target = new THREE.WebGLRenderTarget(Math.max(1, size.x), Math.max(1, size.y), { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, this.target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.85, 0.55, 0.9);
    this.composer.addPass(this.bloom);
    this.finish = new ShaderPass(FinishShader);
    this.composer.addPass(this.finish);
    this.composer.addPass(new OutputPass());
  }

  setColors(colors: AtmosphereColors): void {
    this.colors = colors;
    this.theme = colors.theme;
    const dark = colors.theme === 'dark';
    const b = this.backdrop.material.uniforms;
    (b.uBase.value as THREE.Color).set(colors.background);
    // Neutral tints (violet, indigo), never the sign colours: the backdrop carries no values.
    (b.uTintA.value as THREE.Color).set('#6a55e0');
    (b.uTintB.value as THREE.Color).set('#2f4fb0');
    (b.uGradient.value as THREE.Vector2).set(dark ? 0.5 : 0.95, dark ? 1.65 : 1.02);
    b.uNebula.value = dark ? 1 : 0;

    const f = this.floor.material.uniforms;
    const line = new THREE.Color(colors.grid);
    if (dark) line.lerp(new THREE.Color('#7083a8'), 0.4);
    (f.uLine.value as THREE.Color).copy(line);
    (f.uSpot.value as THREE.Color).set(dark ? '#7f8fc4' : colors.grid);
    f.uOpacity.value = dark ? 0.7 : 0.75;
    f.uSpotStrength.value = dark ? 0.07 : 0.1;

    this.dust.visible = dark;
    this.recolorDust();
    this.bloom.enabled = dark;
    this.finish.uniforms.uVignette.value = dark ? 0.4 : 0.1;
    this.finish.uniforms.uAberration.value = dark ? 0.004 : 0;
  }

  /** Places the floor under the content and spreads the dust around it. */
  setBounds(min: THREE.Vector3, max: THREE.Vector3): void {
    this.center.addVectors(min, max).multiplyScalar(0.5);
    const extent = Math.max(max.x - min.x, max.z - min.z, 2);
    this.radius = Math.max(3, 0.5 * Math.hypot(max.x - min.x, max.y - min.y, max.z - min.z));
    const f = this.floor.material.uniforms;
    (f.uCenter.value as THREE.Vector2).set(this.center.x, this.center.z);
    f.uRadius.value = extent / 2 + 1;
    f.uSpacing.value = extent > 40 ? Math.ceil(extent / 40) : 1;
    this.floor.position.set(this.center.x, min.y - 0.9, this.center.z);
    this.floor.scale.setScalar(extent * 8 + 40);
    this.dust.position.copy(this.center);
    this.dust.scale.setScalar(this.radius);
  }

  setSize(width: number, height: number, pixelRatio: number): void {
    this.composer.setPixelRatio(pixelRatio);
    this.composer.setSize(width, height);
    this.dust.material.uniforms.uScale.value = 3 * pixelRatio * this.radius;
  }

  render(camera: THREE.PerspectiveCamera, pixelRatio: number): void {
    this.backdrop.position.copy(camera.position);
    this.dust.material.uniforms.uScale.value = 3 * pixelRatio * this.radius;
    this.composer.render();
  }

  dispose(): void {
    this.composer.dispose();
    this.target.dispose();
    for (const object of [this.backdrop, this.dust, this.floor]) {
      object.geometry.dispose();
      object.material.dispose();
    }
  }

  /** Dust in a shell around the content (never in front of the data), positions in units of the radius. */
  private createDustGeometry(): THREE.BufferGeometry {
    const positions = new Float32Array(DUST_COUNT * 3);
    const seeds = new Float32Array(DUST_COUNT);
    const sizes = new Float32Array(DUST_COUNT);
    let state = 7;
    const random = () => {
      state = (state * 16807) % 2147483647;
      return state / 2147483647;
    };
    for (let k = 0; k < DUST_COUNT; k++) {
      const z = 2 * random() - 1;
      const phi = 2 * Math.PI * random();
      const s = Math.sqrt(1 - z * z);
      const r = 1.25 + 2.6 * Math.pow(random(), 0.7);
      positions.set([r * s * Math.cos(phi), r * z * 0.75, r * s * Math.sin(phi)], 3 * k);
      seeds[k] = random();
      sizes[k] = 0.5 + 1.6 * Math.pow(random(), 2.2);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(DUST_COUNT * 3), 3));
    return geometry;
  }

  /** Cold and violet specks (never the sign colours), a few bright enough to sparkle. */
  private recolorDust(): void {
    if (!this.colors) return;
    const attribute = this.dust.geometry.getAttribute('aColor') as THREE.BufferAttribute;
    const seeds = this.dust.geometry.getAttribute('aSeed') as THREE.BufferAttribute;
    const cold = new THREE.Color(0.36, 0.46, 0.66);
    const violet = new THREE.Color(0.42, 0.33, 0.78);
    const color = new THREE.Color();
    for (let k = 0; k < attribute.count; k++) {
      const s = seeds.getX(k);
      const pick = (s * 7919) % 1;
      color.copy(pick < 0.22 ? violet : cold);
      if ((s * 104729) % 1 < 0.05) color.multiplyScalar(3.2);
      attribute.setXYZ(k, color.r, color.g, color.b);
    }
    attribute.needsUpdate = true;
  }

  get currentTheme(): ThemeId {
    return this.theme;
  }
}

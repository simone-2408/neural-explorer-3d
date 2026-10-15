/**
 * Shader materials of the Transformer scene.
 *
 * Residual cells: one instanced box per (stage, position, dimension). Each box
 * is a bar of a diverging bar chart — positive values grow up from the slab's
 * mid-line, negative values grow down (so the sign is encoded by direction as
 * well as colour). `uStage` drives the pipeline animation: row s is empty
 * while uStage ≤ s, then shows the value of the row below (the residual
 * stream carries x upward unchanged) and morphs into its own value (the
 * sub-layer *adds* its output): x + t·Δ.
 *
 * Attention arcs: one instanced camera-facing ribbon per (head, query, key),
 * shaped as a quadratic Bézier curve evaluated in the vertex shader. Arcs of
 * layer ℓ grow while row 2ℓ+1 is being computed.
 */
import * as THREE from 'three';
import { FX } from './fx';

const CELL_VERTEX = /* glsl */ `
  attribute float aValue;
  attribute float aPrev;
  // x: stage (row), y: position (column), z: dimension
  attribute vec3 aCell;
  uniform float uStage;
  uniform float uScale;
  uniform float uMaxHeight;
  uniform float uMinHeight;
  uniform vec2 uHover;
  uniform float uFocus;
  uniform float uDimHighlight;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vValue;
  varying float vComputed;
  varying float vHover;
  varying float vFocus;
  varying float vMarked;

  void main() {
    float r = clamp(uStage - aCell.x, 0.0, 1.0);
    float shown = mix(aPrev, aValue, r);
    float t = clamp(abs(shown) * uScale, 0.0, 1.0);
    vValue = shown < 0.0 ? -t : t;
    vComputed = r > 0.0 ? 1.0 : 0.0;
    float extent = vComputed > 0.5 ? mix(uMinHeight, uMaxHeight, pow(t, 0.8)) : uMinHeight;
    // The unit box spans y ∈ [0, 1]: map it to [−min, extent] (positive) or [−extent, min] (negative).
    float y0 = shown >= 0.0 || vComputed < 0.5 ? -uMinHeight : -extent;
    float y1 = shown >= 0.0 || vComputed < 0.5 ? extent : uMinHeight;
    vec3 p = position;
    p.y = mix(y0, y1, position.y);
    vHover = abs(aCell.y - uHover.x) < 0.5 && abs(aCell.x - uHover.y) < 0.5 ? 1.0 : 0.0;
    vFocus = uFocus < 0.0 || abs(aCell.y - uFocus) < 0.5 ? 1.0 : 0.0;
    vMarked = abs(aCell.z - uDimHighlight) < 0.5 ? 1.0 : 0.0;
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
    vViewPosition = -mvPosition.xyz;
    vNormal = normalize(normalMatrix * mat3(instanceMatrix) * normal);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const CELL_FRAGMENT = /* glsl */ `
  uniform vec3 uPositive;
  uniform vec3 uNegative;
  uniform vec3 uNeutral;
  uniform vec3 uEmpty;
  uniform vec3 uBackground;
  uniform vec3 uHighlight;
  uniform float uDim;
  uniform float uGlow;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying float vValue;
  varying float vComputed;
  varying float vHover;
  varying float vFocus;
  varying float vMarked;

  void main() {
    vec3 n = normalize(vNormal);
    vec3 v = normalize(vViewPosition);
    vec3 l = normalize(vec3(0.35, 0.8, 0.55));
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(n, normalize(l + v)), 0.0), 32.0);
    vec3 signColor = vValue < 0.0 ? uNegative : uPositive;
    vec3 base = vComputed > 0.5 ? mix(uNeutral, signColor, pow(abs(vValue), 0.6)) : uEmpty;
    vec3 color = base * (0.42 + 0.68 * diffuse) + vec3(specular * 0.18);
    // Effects: large values emit light.
    color += base * uGlow * 1.0 * smoothstep(0.5, 1.0, abs(vValue)) * vComputed;
    if (vFocus < 0.5) color = mix(color, uBackground, uDim);
    if (vMarked > 0.5 && vComputed > 0.5) color = mix(color, uHighlight, 0.35);
    if (vHover > 0.5) color = mix(color, uHighlight, 0.22);
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;

export function createCellMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: 'ResidualCellMaterial',
    vertexShader: CELL_VERTEX,
    fragmentShader: CELL_FRAGMENT,
    uniforms: {
      uStage: { value: 1e6 },
      uScale: { value: 1 },
      uMaxHeight: { value: 0.55 },
      uMinHeight: { value: 0.025 },
      uHover: { value: new THREE.Vector2(-1, -1) },
      uFocus: { value: -1 },
      uDimHighlight: { value: -1 },
      uPositive: { value: new THREE.Color() },
      uNegative: { value: new THREE.Color() },
      uNeutral: { value: new THREE.Color() },
      uEmpty: { value: new THREE.Color() },
      uBackground: { value: new THREE.Color() },
      uHighlight: { value: new THREE.Color(1, 1, 1) },
      uDim: { value: 0.55 },
      uGlow: FX.uGlow,
    },
  });
}

const ARC_VERTEX = /* glsl */ `
  attribute vec3 aStart;
  attribute vec3 aControl;
  attribute vec3 aEnd;
  // x: width (world units), y: opacity, z: layer, w: emphasis (0 dimmed, 1 normal, 2 highlighted)
  attribute vec4 aStyle;
  attribute vec3 aColor;
  uniform float uStage;
  uniform int uHover;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vSide;
  varying float vAlong;
  varying float vHover;
  varying float vEmphasis;

  vec3 bezier(float t) {
    float u = 1.0 - t;
    return u * u * aStart + 2.0 * u * t * aControl + t * t * aEnd;
  }

  vec3 bezierTangent(float t) {
    return 2.0 * (1.0 - t) * (aControl - aStart) + 2.0 * t * (aEnd - aControl);
  }

  void main() {
    // Arcs of layer l grow during the first half of the computation of row 2l + 1.
    float grow = clamp((uStage - (2.0 * aStyle.z + 1.0)) * 2.0, 0.0, 1.0);
    float t = position.x * grow;
    vec4 mvPosition = modelViewMatrix * vec4(bezier(t), 1.0);
    vec3 tangent = (modelViewMatrix * vec4(bezierTangent(t), 0.0)).xyz;
    vec3 side = cross(tangent, mvPosition.xyz);
    float len = length(side);
    side = len > 1e-6 ? side / len : vec3(0.0);
    vHover = gl_InstanceID == uHover ? 1.0 : 0.0;
    float width = aStyle.x * (vHover > 0.5 ? 1.7 : (aStyle.w > 1.5 ? 1.4 : 1.0));
    mvPosition.xyz += side * position.y * width * 0.5;
    vColor = aColor;
    vAlpha = grow > 0.0 ? aStyle.y : 0.0;
    vSide = position.y;
    vAlong = position.x;
    vEmphasis = aStyle.w;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const ARC_FRAGMENT = /* glsl */ `
  uniform vec3 uHighlight;
  uniform float uDimAlpha;
  uniform float uGlow;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vSide;
  varying float vAlong;
  varying float vHover;
  varying float vEmphasis;

  void main() {
    float edge = 1.0 - smoothstep(0.45, 1.0, abs(vSide));
    float alpha = vAlpha * (0.3 + 0.7 * edge);
    vec3 color = vColor;
    if (vEmphasis < 0.5) alpha *= uDimAlpha;
    if (vEmphasis > 1.5 || vHover > 0.5) {
      color = mix(color, uHighlight, 0.3);
      alpha = max(alpha, 0.85 * (0.3 + 0.7 * edge));
    }
    // Brighter towards the reading token: information flows from the key (start) to the query (end).
    color *= 0.8 + 0.3 * vAlong;
    // Effects: arcs are made of light, as bright as they are opaque (their opacity encodes the weight).
    color *= 1.0 + uGlow * (0.6 + 1.6 * vAlong) * vAlpha * (vEmphasis < 0.5 ? 0.0 : 1.0);
    if (alpha < 0.004) discard;
    gl_FragColor = vec4(color, alpha);
    #include <colorspace_fragment>
  }
`;

export function createArcMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: 'AttentionArcMaterial',
    vertexShader: ARC_VERTEX,
    fragmentShader: ARC_FRAGMENT,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    uniforms: {
      uStage: { value: 1e6 },
      uHover: { value: -1 },
      uHighlight: { value: new THREE.Color(1, 1, 1) },
      uDimAlpha: { value: 0.18 },
      uGlow: FX.uGlow,
    },
  });
}

/** Ribbon in parameter space: x = t ∈ [0, 1] along the curve, y = side ∈ [−1, 1]. */
export function createRibbonGeometry(segments = 28): THREE.InstancedBufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let k = 0; k <= segments; k++) {
    const t = k / segments;
    positions.push(t, -1, 0, t, 1, 0);
    if (k < segments) {
      const a = 2 * k;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

/**
 * Custom shader materials for instanced neurons and connections.
 *
 * Per-instance data lives in InstancedBufferAttributes that mirror the model
 * layout (one entry per neuron / per weight), so updating colours, emphasis,
 * thickness or signal strength is a single buffer upload — no object
 * re-creation. Time-based effects (travelling pulses, reveal of a layer,
 * follow-the-signal flow) are driven by a handful of uniforms.
 *
 * Colours are passed in linear space and converted to the output colour space
 * at the end (`colorspace_fragment`), like three.js built-in materials.
 */
import * as THREE from 'three';

const NEURON_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  // x: emphasis level (0 dim … 3 focus), y: layer index, z: flow score (follow the signal)
  attribute vec3 aState;
  uniform int uHover;
  // x: reveal layer, y: progress of that layer (0..1), z: direction (+1 forward, -1 backward)
  uniform vec3 uReveal;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying vec3 vColor;
  varying float vEmphasis;
  varying float vHover;
  varying float vReveal;
  varying float vCurrent;
  varying float vFlow;

  void main() {
    vColor = aColor;
    vEmphasis = aState.x;
    vFlow = aState.z;
    vHover = gl_InstanceID == uHover ? 1.0 : 0.0;
    float d = (aState.y - uReveal.x) * uReveal.z; // > 0: not reached yet, 0: being computed, < 0: done
    vReveal = d > 0.5 ? 0.0 : (d > -0.5 ? uReveal.y : 1.0);
    vCurrent = abs(d) < 0.5 ? 1.0 : 0.0;
    vec4 mvPosition = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    vNormal = normalize(normalMatrix * mat3(instanceMatrix) * normal);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const NEURON_FRAGMENT = /* glsl */ `
  uniform vec3 uBackground;
  uniform vec3 uHighlight;
  uniform vec3 uEmpty;
  uniform float uDim;
  uniform float uFlowMode;
  uniform vec3 uReveal;
  varying vec3 vNormal;
  varying vec3 vViewPosition;
  varying vec3 vColor;
  varying float vEmphasis;
  varying float vHover;
  varying float vReveal;
  varying float vCurrent;
  varying float vFlow;

  void main() {
    vec3 n = normalize(vNormal);
    vec3 v = normalize(vViewPosition);
    vec3 l = normalize(vec3(0.35, 0.8, 0.55)); // key light, fixed relative to the camera
    float diffuse = max(dot(n, l), 0.0);
    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    float specular = pow(max(dot(n, normalize(l + v)), 0.0), 48.0);
    vec3 base = mix(uEmpty, vColor, vReveal); // value not computed yet → empty neuron
    vec3 color = base * (0.36 + 0.74 * diffuse) + vec3(specular * 0.3) + base * fresnel * 0.4;
    // flash while the layer is being computed
    color += vCurrent * sin(uReveal.y * 3.14159) * 0.22 * uHighlight;
    if (uFlowMode > 0.5) color = mix(color, uBackground, (1.0 - vFlow) * 0.82);
    else if (vEmphasis < 0.5) color = mix(color, uBackground, uDim);
    if (vHover > 0.5) color = mix(color, uHighlight, 0.3);
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;

const CONNECTION_VERTEX = /* glsl */ `
  attribute vec3 aColor;
  // x: emphasis level, y: length (world units), z: 1 = dashed (negative value), w: base opacity
  attribute vec4 aStyle;
  // x: layer index (of the target neuron), y: signed pulse strength (-1..1), z: flow score (0..1)
  attribute vec3 aSignal;
  uniform int uHover;
  uniform vec3 uReveal;
  uniform float uFlowMode;
  uniform vec4 uPulse;
  varying vec3 vColor;
  varying float vEmphasis;
  varying float vAlong;
  varying float vLength;
  varying float vDashed;
  varying float vAlpha;
  varying float vHover;
  varying float vLayer;
  varying float vSignal;
  varying float vFlow;
  varying float vPending;

  void main() {
    vColor = aColor;
    vEmphasis = aStyle.x;
    vLength = aStyle.y;
    vDashed = aStyle.z;
    vAlpha = aStyle.w;
    vLayer = aSignal.x;
    vSignal = aSignal.y;
    vFlow = aSignal.z;
    vHover = gl_InstanceID == uHover ? 1.0 : 0.0;
    vPending = (aSignal.x - uReveal.x) * uReveal.z > 0.5 ? 1.0 : 0.0;
    // The unit cylinder spans y ∈ [-0.5, 0.5]; distance along the connection, in world units:
    vAlong = (position.y + 0.5) * aStyle.y;
    vec3 p = position;
    float thicken = vEmphasis > 2.5 ? 1.9 : (vHover > 0.5 ? 1.6 : (vEmphasis > 1.5 ? 1.2 : 1.0));
    if (uFlowMode > 0.5) thicken *= 0.5 + 1.7 * aSignal.z;
    // The connection carrying a pulse is drawn a little thicker.
    if (uPulse.w > 0.0 && abs(aSignal.x - uPulse.x) < 0.5) thicken *= 1.0 + 0.6 * uPulse.w * abs(aSignal.y);
    p.xz *= thicken;
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(p, 1.0);
  }
`;

const CONNECTION_FRAGMENT = /* glsl */ `
  uniform float uDashSize;
  uniform float uDimAlpha;
  uniform float uAlphaScale;
  uniform vec3 uHighlight;
  uniform vec3 uPositive;
  uniform vec3 uNegative;
  // x: layer carrying the pulse, y: progress 0..1, z: direction (+1 source→target, -1 target→source), w: intensity (0 = off)
  uniform vec4 uPulse;
  uniform float uFlowMode;
  uniform float uTime;
  varying vec3 vColor;
  varying float vEmphasis;
  varying float vAlong;
  varying float vLength;
  varying float vDashed;
  varying float vAlpha;
  varying float vHover;
  varying float vLayer;
  varying float vSignal;
  varying float vFlow;
  varying float vPending;

  void main() {
    // Negative values are dashed: the sign is never encoded by colour alone.
    if (vDashed > 0.5 && fract(vAlong / uDashSize) > 0.55) discard;
    float u = clamp(vAlong / max(vLength, 1e-4), 0.0, 1.0);
    float alpha = vAlpha * uAlphaScale;
    vec3 color = vColor;
    if (vEmphasis < 0.5) {
      alpha *= uDimAlpha;
    } else if (vEmphasis > 2.5) {
      alpha = 1.0;
      color = mix(color, uHighlight, 0.25);
    } else if (vEmphasis > 1.5) {
      alpha = max(vAlpha, 0.85);
    }
    // Connections the signal has not reached yet are faded.
    alpha *= mix(1.0, 0.16, vPending);

    // A pulse travelling along the connections of one layer.
    if (uPulse.w > 0.0 && abs(vLayer - uPulse.x) < 0.5) {
      float center = uPulse.z > 0.0 ? uPulse.y : 1.0 - uPulse.y;
      float glow = exp(-pow((u - center) / 0.09, 2.0)) * uPulse.w;
      float strength = abs(vSignal);
      vec3 pulseColor = mix(vSignal >= 0.0 ? uPositive : uNegative, vec3(1.0), 0.45);
      color = mix(color, pulseColor, glow * (0.45 + 0.55 * strength));
      alpha = max(alpha, glow * (0.35 + 0.65 * strength));
    }

    // Follow the signal: opacity from the flow score, plus pulses that run along the path.
    if (uFlowMode > 0.5) {
      alpha = mix(0.03, 1.0, pow(vFlow, 0.6));
      float center = fract(uTime * 0.45 - vLayer * 0.3);
      float glow = exp(-pow((u - center) / 0.06, 2.0)) * vFlow;
      color = mix(color, mix(color, vec3(1.0), 0.65), glow);
    }

    if (vHover > 0.5) {
      alpha = 1.0;
      color = mix(color, uHighlight, 0.3);
    }
    gl_FragColor = vec4(color, alpha);
    #include <colorspace_fragment>
  }
`;

/** Reveal state that shows every layer (no forward/backward animation in progress). */
export const REVEAL_ALL = new THREE.Vector3(1e6, 1, 1);

export function createNeuronMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: 'NeuronMaterial',
    vertexShader: NEURON_VERTEX,
    fragmentShader: NEURON_FRAGMENT,
    uniforms: {
      uHover: { value: -1 },
      uReveal: { value: REVEAL_ALL.clone() },
      uBackground: { value: new THREE.Color() },
      uHighlight: { value: new THREE.Color(1, 1, 1) },
      uEmpty: { value: new THREE.Color() },
      uDim: { value: 0.72 },
      uFlowMode: { value: 0 },
    },
  });
}

export function createConnectionMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: 'ConnectionMaterial',
    vertexShader: CONNECTION_VERTEX,
    fragmentShader: CONNECTION_FRAGMENT,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uHover: { value: -1 },
      uReveal: { value: REVEAL_ALL.clone() },
      uDashSize: { value: 0.2 },
      uDimAlpha: { value: 0.12 },
      uAlphaScale: { value: 1 },
      uHighlight: { value: new THREE.Color(1, 1, 1) },
      uPositive: { value: new THREE.Color() },
      uNegative: { value: new THREE.Color() },
      uPulse: { value: new THREE.Vector4(-1, 0, 1, 0) },
      uFlowMode: { value: 0 },
      uTime: { value: 0 },
    },
  });
}

/**
 * Visual effects shared by every scene (glow, ambient motion): one set of
 * uniforms referenced by all the materials, so switching the effects on or off,
 * or advancing the ambient clock, needs no material rebuild.
 *
 * Glow is HDR light: materials add an emissive term (above 1 in linear space)
 * to elements that carry a large value — strongly active neurons and feature
 * maps, large residual values, the pulses of a forward pass — and the bloom
 * pass of the Atmosphere spreads only that light. Nothing glows that does not
 * encode a value, and the colours stay those of the palette.
 */
import * as THREE from 'three';

export const FX = {
  /** 0 = classic rendering, 1 = glow on (dark theme with effects enabled). */
  uGlow: { value: 0 },
  /** 1 while the ambient animation runs (effects on, motion not reduced), else 0. */
  uMotion: { value: 0 },
  /** Ambient clock, in seconds. */
  uTime: { value: 0 },
};

/** GLSL: chroma of a linear colour — 0 for greys (values near zero), up to 1 for saturated sign colours. */
export const FX_CHROMA = /* glsl */ `
  float fxChroma(vec3 c) {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
  }
`;

/**
 * Adds value-driven glow to a built-in lit material coloured per instance or
 * per vertex: the more saturated the colour (the further its value from zero),
 * the more light it emits. `strength` scales the emitted light, `from` is the
 * chroma below which nothing glows.
 */
export function withGlow<T extends THREE.MeshLambertMaterial | THREE.MeshStandardMaterial>(material: T, strength = 1.2, from = 0.45): T {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uGlow = FX.uGlow;
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\nuniform float uGlow;\n${FX_CHROMA}`)
      .replace(
        '#include <emissivemap_fragment>',
        `#include <emissivemap_fragment>
        #ifdef USE_COLOR
          totalEmissiveRadiance += vColor.rgb * uGlow * ${strength.toFixed(3)} * smoothstep(${from.toFixed(3)}, 0.95, fxChroma(vColor.rgb));
        #endif`,
      );
  };
  material.customProgramCacheKey = () => `fx-glow-${strength}-${from}`;
  return material;
}

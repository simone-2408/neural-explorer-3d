import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { CAMERA_VIEWS, CameraController, FIT_MARGIN } from './CameraController';
import type { Viewport } from './Viewport';

/** The parts of Viewport the controller uses, without WebGL. */
function fakeViewport(width: number, height: number) {
  const camera = new THREE.PerspectiveCamera(42, width / height, 0.05, 2000);
  const controls = { target: new THREE.Vector3(), addEventListener: () => undefined, update: () => false };
  const viewport = {
    camera,
    controls,
    size: { width, height },
    animations: { cancelChannel: () => undefined },
    onResize: () => () => undefined,
    setContentBounds: () => undefined,
    requestRender: () => undefined,
  };
  return { camera, controls, controller: new CameraController(viewport as unknown as Viewport) };
}

/** Screen extents (NDC) of the eight corners of a box. */
function projectedExtents(camera: THREE.PerspectiveCamera, min: THREE.Vector3, max: THREE.Vector3) {
  camera.updateMatrixWorld();
  const p = new THREE.Vector3();
  const e = { left: Infinity, right: -Infinity, bottom: Infinity, top: -Infinity };
  for (let k = 0; k < 8; k++) {
    p.set(k & 1 ? max.x : min.x, k & 2 ? max.y : min.y, k & 4 ? max.z : min.z).project(camera);
    e.left = Math.min(e.left, p.x);
    e.right = Math.max(e.right, p.x);
    e.bottom = Math.min(e.bottom, p.y);
    e.top = Math.max(e.top, p.y);
  }
  return e;
}

describe('CameraController.fitDirection', () => {
  // A long, flat box like two stacked networks, off the origin.
  const min = new THREE.Vector3(-14, -9, -2.5);
  const max = new THREE.Vector3(15, 8, 2.5);
  const padding = 28;

  for (const [width, height] of [
    [465, 805],
    [1040, 805],
  ]) {
    it(`with a padding, frames the box exactly and centred in every preset view (${width}×${height})`, () => {
      const { camera, controller } = fakeViewport(width, height);
      // Largest NDC extent on each axis: the padding in pixels, then the margin.
      const limitX = (1 - (2 * padding) / width) / FIT_MARGIN;
      const limitY = (1 - (2 * padding) / height) / FIT_MARGIN;
      controller.setBounds(min.toArray(), max.toArray(), padding);
      for (const view of CAMERA_VIEWS) {
        controller.setView(view.id, false);
        const e = projectedExtents(camera, min, max);
        // Inside the frustum, with the room kept for the HTML labels…
        expect(Math.max(-e.left, e.right) / limitX).toBeLessThanOrEqual(1 + 1e-6);
        expect(Math.max(-e.bottom, e.top) / limitY).toBeLessThanOrEqual(1 + 1e-6);
        // …touching it on the limiting axis, and centred on both.
        expect(Math.max(e.right / limitX, e.top / limitY)).toBeCloseTo(1, 4);
        expect(e.left + e.right).toBeCloseTo(0, 4);
        expect(e.bottom + e.top).toBeCloseTo(0, 4);
      }
    });
  }

  it('without a padding, aims at the box centre with the whole box in view', () => {
    const { camera, controls, controller } = fakeViewport(750, 805);
    controller.setBounds(min.toArray(), max.toArray());
    controller.reset(false);
    expect(controls.target.toArray()).toEqual([0.5, -0.5, 0]);
    const e = projectedExtents(camera, min, max);
    for (const v of [e.left, e.bottom]) expect(v).toBeGreaterThan(-1);
    for (const v of [e.right, e.top]) expect(v).toBeLessThan(1);
  });
});

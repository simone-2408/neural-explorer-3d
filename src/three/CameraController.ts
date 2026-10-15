/**
 * High-level camera operations on top of OrbitControls: preset views,
 * reset, focus on an object, face a layer — all as smooth flights driven by
 * the AnimationController ("camera" channel).
 */
import * as THREE from 'three';
import { easeInOutCubic } from '../animation/easing';
import type { Vec3 } from '../visualization/layout';
import type { Viewport } from './Viewport';

export type CameraView = 'iso' | 'side' | 'top' | 'front';

export const CAMERA_VIEWS: { id: CameraView; label: string; key: string; description: string }[] = [
  { id: 'iso', label: '3D', key: '1', description: 'Three-quarter perspective view' },
  { id: 'side', label: 'Side', key: '2', description: 'Side view: the classic left-to-right diagram' },
  { id: 'top', label: 'Top', key: '3', description: 'Top view, looking down on the layers' },
  { id: 'front', label: 'Front', key: '4', description: 'Along the signal: from the input towards the output' },
];

/** Direction from the target to the camera for each preset view (layers run along +X). */
const VIEW_DIRECTIONS: Record<CameraView, Vec3> = {
  iso: [-0.45, 0.4, 1],
  side: [0, 0.0001, 1],
  top: [0, 1, 0.0001],
  front: [-1, 0.16, 0.24],
};

export class CameraController {
  private readonly viewport: Viewport;
  private readonly box = new THREE.Box3(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(5, 5, 5));
  /** Direction of the view currently framing the content; null once the user moves the camera. */
  private fitted: Vec3 | null = null;

  constructor(viewport: Viewport) {
    this.viewport = viewport;
    viewport.controls.addEventListener('start', () => (this.fitted = null));
    // Keep a preset view framed when the drawing area changes size (window, side panels).
    viewport.onResize(() => {
      if (this.fitted) this.fitDirection(this.fitted, false);
    });
  }

  /** Bounding box of the content, used by reset and preset views. */
  setBounds(min: Vec3, max: Vec3): void {
    this.box.min.set(min[0], min[1], min[2]);
    this.box.max.set(max[0], max[1], max[2]);
    this.viewport.setContentBounds(this.box.min, this.box.max);
  }

  /**
   * Smallest camera distance (along `direction`, from `target`) at which all
   * eight corners of the content box are inside the view frustum.
   */
  private fitBoxDistance(direction: THREE.Vector3, target: THREE.Vector3): number {
    const camera = this.viewport.camera;
    // HTML labels extend beyond the 3D box: keep more room on narrow (portrait) viewports.
    const margin = camera.aspect < 1 ? 1.16 : 1.06;
    const forward = direction.clone().normalize();
    const upHint = Math.abs(forward.y) > 0.99 ? new THREE.Vector3(0, 0, -1) : new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(upHint, forward).normalize();
    const up = new THREE.Vector3().crossVectors(forward, right).normalize();
    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const tanH = tanV * camera.aspect;
    const { min, max } = this.box;
    const p = new THREE.Vector3();
    let distance = 0;
    for (let k = 0; k < 8; k++) {
      p.set(k & 1 ? max.x : min.x, k & 2 ? max.y : min.y, k & 4 ? max.z : min.z).sub(target);
      const z = p.dot(forward);
      distance = Math.max(distance, z + Math.abs(p.dot(right)) / tanH, z + Math.abs(p.dot(up)) / tanV);
    }
    return distance * margin;
  }

  /** Camera distance at which a sphere of `radius` fills the view. */
  fitDistance(radius: number, margin = 1): number {
    const camera = this.viewport.camera;
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    return (radius * margin) / Math.sin(Math.min(vFov, hFov) / 2);
  }

  reset(animate = true): void {
    this.setView('iso', animate);
  }

  setView(view: CameraView, animate = true): void {
    this.fitDirection(VIEW_DIRECTIONS[view], animate);
  }

  /** Frames the content box, looking at it from `direction` (from the target towards the camera). */
  fitDirection(direction: Vec3, animate = true): void {
    const dir = new THREE.Vector3(...direction).normalize();
    const center = this.box.getCenter(new THREE.Vector3());
    const position = center.clone().addScaledVector(dir, this.fitBoxDistance(dir, center));
    this.flyTo(position, center, animate);
    this.fitted = direction;
  }

  /** Centres a sphere in the view, keeping the current viewing direction. */
  focus(center: Vec3, radius: number, animate = true): void {
    this.fitted = null;
    const camera = this.viewport.camera;
    const target = new THREE.Vector3(...center);
    const direction = camera.position.clone().sub(this.viewport.controls.target);
    if (direction.lengthSq() < 1e-8) direction.set(0, 0, 1);
    direction.normalize();
    const position = target.clone().addScaledVector(direction, this.fitDistance(Math.max(radius, 0.4), 1.1));
    this.flyTo(position, target, animate);
  }

  /** Looks at a layer face-on (perpendicular to its plane), from the input side. */
  faceLayer(center: Vec3, halfWidth: number, halfHeight: number, animate = true): void {
    this.fitted = null;
    const target = new THREE.Vector3(...center);
    const direction = new THREE.Vector3(-1, 0.12, 0.1).normalize();
    const position = target.clone().addScaledVector(direction, this.fitDistance(Math.hypot(halfWidth, halfHeight), 1.02));
    this.flyTo(position, target, animate);
  }

  /**
   * Smooth flight: interpolates the orbit target linearly and the camera
   * offset in spherical coordinates (so the camera orbits instead of cutting
   * through the scene).
   */
  flyTo(position: THREE.Vector3, target: THREE.Vector3, animate = true): void {
    const { camera, controls, animations } = this.viewport;
    animations.cancelChannel('camera');
    if (!animate) {
      controls.target.copy(target);
      camera.position.copy(position);
      camera.lookAt(target);
      controls.update();
      this.viewport.requestRender();
      return;
    }

    const startTarget = controls.target.clone();
    const from = new THREE.Spherical().setFromVector3(camera.position.clone().sub(startTarget));
    const to = new THREE.Spherical().setFromVector3(position.clone().sub(target));
    let dTheta = to.theta - from.theta;
    if (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    if (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    const current = new THREE.Spherical();
    const radiusRatio = to.radius / Math.max(from.radius, 1e-6);

    // Damping would add leftover momentum on top of the flight.
    const damping = controls.enableDamping;
    controls.enableDamping = false;
    const restore = () => {
      controls.enableDamping = damping;
    };

    animations.tween({
      channel: 'camera',
      duration: 0.85,
      easing: easeInOutCubic,
      onUpdate: (t) => {
        current.radius = from.radius * Math.pow(radiusRatio, t);
        current.phi = from.phi + (to.phi - from.phi) * t;
        current.theta = from.theta + dTheta * t;
        controls.target.lerpVectors(startTarget, target, t);
        camera.position.setFromSpherical(current).add(controls.target);
        camera.lookAt(controls.target);
        this.viewport.requestRender();
      },
      onComplete: restore,
      onCancel: restore,
    });
  }
}

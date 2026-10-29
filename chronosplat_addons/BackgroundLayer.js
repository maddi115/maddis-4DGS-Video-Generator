/**
 * BackgroundLayer.js - one still store per segment, moved into each frame's camera.
 * manifest.background = { files: [...], segments: [...], poses: [[scale, qx, qy, qz, qw, px, py, pz], ...] }
 * People frames are never moved; only this store follows the camera (poses are optional).
 */
import { SplatMesh } from '@sparkjsdev/spark';
import { Quaternion } from 'three';

const FLIP = new Quaternion(1, 0, 0, 0);
const NOFLIP = new Quaternion(0, 0, 0, 1);
const _q = new Quaternion();

export class BackgroundLayer {
  constructor({ scene, base, background, flipX = true }) {
    this.scene = scene;
    this.flipX = flipX;
    this.enabled = Boolean(background?.files?.length && background?.segments?.length);
    this.urls = this.enabled ? background.files.map((f) => new URL(f, base).href) : [];
    this.segments = this.enabled ? background.segments : [];
    this.poses = this.enabled && Array.isArray(background.poses) ? background.poses : null;
    this.meshes = new Map();
    this.failed = new Set();
    this.visibleSeg = -1;
  }

  segOf(frameIndex) { return this.segments[frameIndex] ?? 0; }

  isReadyFor(frameIndex) {
    if (!this.enabled) return true;
    if (this.failed.has(this.segOf(frameIndex))) return true;
    return this.meshes.get(this.segOf(frameIndex))?.ready === true;
  }

  _load(seg) {
    if (seg < 0 || seg >= this.urls.length) return;
    if (this.meshes.has(seg) || this.failed.has(seg)) return;
    const mesh = new SplatMesh({ url: this.urls[seg] });
    const entry = { mesh, ready: false };
    this.meshes.set(seg, entry);
    mesh.visible = false;
    mesh.initialized
      .then(() => {
        if (this.meshes.get(seg) !== entry) { mesh.dispose(); return; }
        if (this.flipX) mesh.quaternion.set(1, 0, 0, 0);
        this.scene.add(mesh);
        entry.ready = true;
      })
      .catch((err) => {
        console.warn(`[BackgroundLayer] background ${seg + 1} failed: ${this.urls[seg]}`, err);
        this.meshes.delete(seg);
        this.failed.add(seg);
        try { mesh.dispose(); } catch { }
      });
  }

  update(activeIndex, shownIndex = activeIndex) {
    if (!this.enabled) return;
    const want = this.segOf(activeIndex);
    const show = this.segOf(shownIndex >= 0 ? shownIndex : activeIndex);
    this._load(show);
    this._load(want);
    this._load((want + 1) % this.urls.length);
    const e = this.meshes.get(show);
    if (e?.ready && this.visibleSeg !== show) {
      for (const [s, x] of this.meshes) if (x.ready) x.mesh.visible = s === show;
      this.visibleSeg = show;
    }
    const vis = this.meshes.get(this.visibleSeg);
    if (vis?.ready) this._applyPose(vis.mesh, this.poses?.[shownIndex >= 0 ? shownIndex : activeIndex]);
    const keep = new Set([show, want, (want + 1) % this.urls.length, this.visibleSeg]);
    for (const [s, x] of [...this.meshes]) {
      if (keep.has(s)) continue;
      if (x.ready) this.scene.remove(x.mesh);
      try { x.mesh.dispose(); } catch { }
      this.meshes.delete(s);
    }
  }

  _applyPose(mesh, p) {
    const flip = this.flipX ? FLIP : NOFLIP;
    if (!p) { mesh.quaternion.copy(flip); mesh.position.set(0, 0, 0); mesh.scale.setScalar(1); return; }
    _q.set(p[1], p[2], p[3], p[4]);
    mesh.quaternion.copy(flip).multiply(_q);
    mesh.position.set(p[5], p[6], p[7]).applyQuaternion(flip);
    mesh.scale.setScalar(p[0]);
  }

  setFlipX(value) {
    this.flipX = Boolean(value);
    for (const [, x] of this.meshes) {
      if (!x.ready) continue;
      if (this.flipX) x.mesh.quaternion.set(1, 0, 0, 0);
      else x.mesh.quaternion.set(0, 0, 0, 1);
    }
  }

  dispose() {
    for (const [, x] of this.meshes) {
      if (x.ready) this.scene.remove(x.mesh);
      try { x.mesh.dispose(); } catch { }
    }
    this.meshes.clear();
    this.visibleSeg = -1;
  }
}

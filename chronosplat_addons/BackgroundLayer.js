/**
 * BackgroundLayer.js â€” a still background shared by many frames.
 *
 * Sequences built with build_layers.py store, per segment of frames, ONE
 * background splat file (everything that does not move) plus a small
 * per-frame file with only the moving splats. FrameCache streams the moving
 * frames as usual; this class keeps the matching background on screen.
 *
 * manifest.background = {
 *   files:    ["bg/frame_0001.sog", ...],   // one per segment
 *   segments: [0, 0, 0, ..., 1, 1, ...],    // segment index for every frame
 * }
 *
 * Inert when the manifest has no background block, so ordinary sequences are
 * unaffected.
 */

import { SplatMesh } from '@sparkjsdev/spark';

export class BackgroundLayer {
  constructor({ scene, base, background, flipX = true }) {
    this.scene = scene;
    this.flipX = flipX;
    this.enabled = Boolean(background?.files?.length && background?.segments?.length);
    this.urls = this.enabled ? background.files.map((f) => new URL(f, base).href) : [];
    this.segments = this.enabled ? background.segments : [];
    /** @type {Map<number, {mesh: SplatMesh, ready: boolean}>} */
    this.meshes = new Map();
    this.failed = new Set();
    this.visibleSeg = -1;
  }

  segOf(frameIndex) {
    return this.segments[frameIndex] ?? 0;
  }

  /** True when the background a frame needs is decoded (always true if unused). */
  isReadyFor(frameIndex) {
    if (!this.enabled) return true;
    if (this.failed.has(this.segOf(frameIndex))) return true; // do not stall forever
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
        try { mesh.dispose(); } catch { /* already gone */ }
      });
  }

  /**
   * Call once per rendered frame.
   * @param {number} activeIndex  frame the playhead is on (preload for it)
   * @param {number} shownIndex   frame actually on screen (match its background)
   */
  update(activeIndex, shownIndex = activeIndex) {
    if (!this.enabled) return;
    const want = this.segOf(activeIndex);
    const show = this.segOf(shownIndex >= 0 ? shownIndex : activeIndex);
    this._load(show);
    this._load(want);
    this._load((want + 1) % this.urls.length); // next segment (wraps around when looping)

    const e = this.meshes.get(show);
    if (e?.ready && this.visibleSeg !== show) {
      for (const [s, x] of this.meshes) if (x.ready) x.mesh.visible = s === show;
      this.visibleSeg = show;
    }

    // Keep at most: shown, wanted and the next one. Free the rest.
    const keep = new Set([show, want, (want + 1) % this.urls.length, this.visibleSeg]);
    for (const [s, x] of [...this.meshes]) {
      if (keep.has(s)) continue;
      if (x.ready) this.scene.remove(x.mesh);
      try { x.mesh.dispose(); } catch { /* already gone */ }
      this.meshes.delete(s);
    }
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
      try { x.mesh.dispose(); } catch { /* already gone */ }
    }
    this.meshes.clear();
    this.visibleSeg = -1;
  }
}

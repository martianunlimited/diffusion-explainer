/**
 * Cross-Attention Spatial Heatmap Component
 * Overlays token-specific attention maps on top of the decoded image
 * and the latent canvas, visualizing where text tokens guide visual features.
 */

import { computeAttentionHeatmap } from '../utils/math.js';
import { COLORMAPS, getRGB } from '../utils/colormaps.js';

export class CrossAttentionManager {
  constructor({ canvasId, colormap = COLORMAPS.TURBO }) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.colormap = colormap;
    this.alpha = 0.55;
    this.enabled = true;
    this.currentHotspot = null;
    this.currentToken = null;

    this.offCanvas = document.createElement('canvas');
    this.offCanvas.width = 64;
    this.offCanvas.height = 64;
    this.offCtx = this.offCanvas.getContext('2d');
  }

  setColormap(cmap) {
    this.colormap = cmap;
    this.render();
  }

  setAlpha(alpha) {
    this.alpha = Math.max(0, Math.min(1, alpha));
    this.render();
  }

  setEnabled(enabled) {
    this.enabled = !!enabled;
    this.render();
  }

  setTokenAttention(token, hotspot) {
    this.currentToken = token;
    this.currentHotspot = hotspot;
    this.render();
    this.updateStats();
  }

  render() {
    if (!this.canvas || !this.ctx) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (!this.enabled || !this.currentHotspot) return;

    // Generate 64x64 attention map
    const map = computeAttentionHeatmap(this.currentHotspot, 64, 64);
    const imgData = this.offCtx.createImageData(64, 64);

    for (let i = 0; i < map.length; i++) {
      const val = map[i];
      const [r, g, b] = getRGB(val, this.colormap);

      const px = i * 4;
      imgData.data[px] = r;
      imgData.data[px + 1] = g;
      imgData.data[px + 2] = b;
      // Alpha mapped smoothly
      imgData.data[px + 3] = Math.round(Math.pow(val, 0.7) * 255);
    }

    this.offCtx.putImageData(imgData, 0, 0);

    // Upscale smoothly with bicubic-like interpolation onto canvas
    this.ctx.save();
    this.ctx.globalAlpha = this.alpha;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(this.offCanvas, 0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }

  updateStats() {
    const statsElem = document.getElementById('attention-stats');
    if (!statsElem) return;

    if (!this.currentHotspot || !this.currentToken) {
      statsElem.innerHTML = '<span class="att-hint">No token selected. Click any highlighted token in the prompt above.</span>';
      return;
    }

    const { cx, cy, rx, ry, intensity } = this.currentHotspot;
    const peakX = Math.round(cx * 100);
    const peakY = Math.round(cy * 100);
    const area = Math.round(Math.PI * (rx * 100) * (ry * 100) / 4);

    statsElem.innerHTML = `
      <div class="att-stat-item">
        <span class="lbl">Focus Token</span>
        <span class="val accent-cyan">"${this.currentToken}"</span>
      </div>
      <div class="att-stat-item">
        <span class="lbl">Peak Spatial Centroid</span>
        <span class="val">(${peakX}%, ${peakY}%)</span>
      </div>
      <div class="att-stat-item">
        <span class="lbl">Attention Density</span>
        <span class="val">${Math.round(intensity * 100)}%</span>
      </div>
      <div class="att-stat-item">
        <span class="lbl">Receptive Coverage</span>
        <span class="val">~${area}% of frame</span>
      </div>
    `;
  }
}

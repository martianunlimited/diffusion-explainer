/**
 * 64×64×4 Latent Space Visualizer Component
 * Supports PCA-to-RGB projection, individual channel views (z0..z3),
 * 2×2 channel matrix, and live coordinate/tensor inspection.
 */

import { projectLatentToRGB, generateLatentTensor } from '../utils/math.js';
import { COLORMAPS, getRGB } from '../utils/colormaps.js';

export class LatentRenderer {
  constructor({ canvasId, statsId, colormap = COLORMAPS.TURBO }) {
    this.canvas = document.getElementById(canvasId);
    this.statsContainer = document.getElementById(statsId);
    this.colormap = colormap;
    this.viewMode = 'pca'; // 'pca' | 'ch0' | 'ch1' | 'ch2' | 'ch3' | 'grid'
    this.currentLatent = null;
    this.ctx = this.canvas ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
    this.offCanvas = document.createElement('canvas');
    this.offCanvas.width = 64;
    this.offCanvas.height = 64;
    this.offCtx = this.offCanvas.getContext('2d', { willReadFrequently: true });
    
    this.bindHover();
  }

  setColormap(cmap) {
    this.colormap = cmap;
    if (this.currentLatent) {
      this.render();
    }
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (this.currentLatent) {
      this.render();
    }
  }

  updateLatents(clean64ImageData, step, totalSteps = 50, seed = 42) {
    this.currentLatent = generateLatentTensor(clean64ImageData, step, totalSteps, seed);
    this.render();
    this.updateStats();
  }

  render() {
    if (!this.canvas || !this.ctx || !this.currentLatent) return;

    const { width, height, data } = this.currentLatent;
    const planeSize = width * height;

    if (this.viewMode === 'pca') {
      this.renderPCA(data, width, height, planeSize);
    } else if (this.viewMode.startsWith('ch')) {
      const channelIdx = parseInt(this.viewMode.replace('ch', ''), 10);
      this.renderSingleChannel(data, channelIdx, width, height, planeSize);
    } else if (this.viewMode === 'grid') {
      this.render2x2Grid(data, width, height, planeSize);
    }
  }

  renderPCA(data, width, height, planeSize) {
    const imgData = this.offCtx.createImageData(width, height);

    for (let i = 0; i < planeSize; i++) {
      const z0 = data[i];
      const z1 = data[planeSize + i];
      const z2 = data[2 * planeSize + i];
      const z3 = data[3 * planeSize + i];

      const [r, g, b] = projectLatentToRGB(z0, z1, z2, z3);

      const px = i * 4;
      imgData.data[px] = r;
      imgData.data[px + 1] = g;
      imgData.data[px + 2] = b;
      imgData.data[px + 3] = 255;
    }

    this.offCtx.putImageData(imgData, 0, 0);

    // Upscale smoothly or pixelated onto display canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.imageSmoothingEnabled = false; // Crisp latent pixels
    this.ctx.drawImage(this.offCanvas, 0, 0, this.canvas.width, this.canvas.height);
  }

  renderSingleChannel(data, ch, width, height, planeSize) {
    const imgData = this.offCtx.createImageData(width, height);
    const offset = ch * planeSize;

    // Find min/max for channel normalization
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < planeSize; i++) {
      const v = data[offset + i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

    for (let i = 0; i < planeSize; i++) {
      const raw = data[offset + i];
      const norm = (raw - minVal) / range;
      const [r, g, b] = getRGB(norm, this.colormap);

      const px = i * 4;
      imgData.data[px] = r;
      imgData.data[px + 1] = g;
      imgData.data[px + 2] = b;
      imgData.data[px + 3] = 255;
    }

    this.offCtx.putImageData(imgData, 0, 0);

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.offCanvas, 0, 0, this.canvas.width, this.canvas.height);
  }

  render2x2Grid(data, width, height, planeSize) {
    const halfW = this.canvas.width / 2;
    const halfH = this.canvas.height / 2;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (let ch = 0; ch < 4; ch++) {
      const gx = (ch % 2) * halfW;
      const gy = Math.floor(ch / 2) * halfH;

      const imgData = this.offCtx.createImageData(width, height);
      const offset = ch * planeSize;

      let minVal = Infinity, maxVal = -Infinity;
      for (let i = 0; i < planeSize; i++) {
        const v = data[offset + i];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      const range = maxVal - minVal > 1e-4 ? maxVal - minVal : 1.0;

      for (let i = 0; i < planeSize; i++) {
        const norm = (data[offset + i] - minVal) / range;
        const [r, g, b] = getRGB(norm, this.colormap);
        const px = i * 4;
        imgData.data[px] = r;
        imgData.data[px + 1] = g;
        imgData.data[px + 2] = b;
        imgData.data[px + 3] = 255;
      }

      this.offCtx.putImageData(imgData, 0, 0);
      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(this.offCanvas, gx, gy, halfW, halfH);

      // Label badge
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      this.ctx.fillRect(gx + 6, gy + 6, 44, 18);
      this.ctx.fillStyle = '#00f5d4';
      this.ctx.font = '600 10px monospace';
      this.ctx.fillText(`Ch ${ch}`, gx + 10, gy + 19);

      // Border line
      this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      this.ctx.strokeRect(gx, gy, halfW, halfH);
    }
  }

  updateStats() {
    if (!this.statsContainer || !this.currentLatent) return;

    const data = this.currentLatent.data;
    let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
    const len = data.length;

    for (let i = 0; i < len; i++) {
      const v = data[i];
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const mean = sum / len;
    const variance = Math.max(0, sumSq / len - mean * mean);
    const std = Math.sqrt(variance);

    this.statsContainer.innerHTML = `
      <div class="stat-pill"><span class="lbl">Tensor</span><span class="val">1×4×64×64</span></div>
      <div class="stat-pill"><span class="lbl">Mean μ</span><span class="val">${mean.toFixed(3)}</span></div>
      <div class="stat-pill"><span class="lbl">Std σ</span><span class="val">${std.toFixed(3)}</span></div>
      <div class="stat-pill"><span class="lbl">Range</span><span class="val">[${min.toFixed(2)}, ${max.toFixed(2)}]</span></div>
    `;
  }

  bindHover() {
    if (!this.canvas) return;

    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.currentLatent) return;
      const rect = this.canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      const lx = Math.floor(nx * 64);
      const ly = Math.floor(ny * 64);
      if (lx >= 0 && lx < 64 && ly >= 0 && ly < 64) {
        const idx = ly * 64 + lx;
        const pSize = 4096;
        const z0 = this.currentLatent.data[idx];
        const z1 = this.currentLatent.data[pSize + idx];
        const z2 = this.currentLatent.data[2 * pSize + idx];
        const z3 = this.currentLatent.data[3 * pSize + idx];

        const readout = document.getElementById('latent-hover-readout');
        if (readout) {
          readout.innerHTML = `
            <strong>Latent (${lx}, ${ly})</strong>: 
            z₀=${z0.toFixed(2)} | z₁=${z1.toFixed(2)} | z₂=${z2.toFixed(2)} | z₃=${z3.toFixed(2)}
          `;
        }
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      const readout = document.getElementById('latent-hover-readout');
      if (readout) {
        readout.textContent = 'Hover over the 64×64 latent canvas to inspect individual tensor values.';
      }
    });
  }
}

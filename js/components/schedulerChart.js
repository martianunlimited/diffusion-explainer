/**
 * Noise Scheduler Curves & SNR Interactive SVG Chart Component
 * Visualizes alpha_bar (signal), sigma (noise), and SNR across timesteps
 * with a real-time tracking scrubber cursor and click-to-seek functionality.
 */

import { getScheduleParams } from '../utils/math.js';

export class SchedulerChart {
  constructor({ containerId, onSeekStep }) {
    this.container = document.getElementById(containerId);
    this.onSeekStep = onSeekStep || (() => {});
    this.currentStep = 50;
    this.totalSteps = 50;

    this.init();
  }

  setStep(step) {
    this.currentStep = Math.max(0, Math.min(this.totalSteps, step));
    this.updateCursor();
  }

  init() {
    if (!this.container) return;

    const w = 480;
    const h = 140;
    const padL = 40;
    const padR = 20;
    const padT = 15;
    const padB = 25;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Generate curve paths
    const alphaPoints = [];
    const sigmaPoints = [];

    for (let s = 0; s <= this.totalSteps; s++) {
      // Step 50 is left (noise), step 0 is right (clean)
      const x = padL + ((this.totalSteps - s) / this.totalSteps) * plotW;
      const params = getScheduleParams(s, this.totalSteps);

      const yAlpha = padT + (1.0 - params.alphaBar) * plotH;
      const ySigma = padT + (1.0 - params.sigma) * plotH;

      alphaPoints.push(`${x.toFixed(1)},${yAlpha.toFixed(1)}`);
      sigmaPoints.push(`${x.toFixed(1)},${ySigma.toFixed(1)}`);
    }

    this.plotW = plotW;
    this.plotH = plotH;
    this.padL = padL;
    this.padT = padT;

    this.container.innerHTML = `
      <div class="scheduler-chart-wrap">
        <svg viewBox="0 0 ${w} ${h}" class="scheduler-svg" id="scheduler-svg">
          <!-- Background Grid Lines -->
          <line x1="${padL}" y1="${padT}" x2="${w - padR}" y2="${padT}" stroke="rgba(255,255,255,0.06)" />
          <line x1="${padL}" y1="${padT + plotH * 0.5}" x2="${w - padR}" y2="${padT + plotH * 0.5}" stroke="rgba(255,255,255,0.06)" />
          <line x1="${padL}" y1="${padT + plotH}" x2="${w - padR}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.12)" />

          <!-- Alpha_bar Curve (Signal strength) -->
          <path d="M ${alphaPoints.join(' L ')}" fill="none" stroke="#00f5d4" stroke-width="2.5" />
          
          <!-- Sigma Curve (Noise strength) -->
          <path d="M ${sigmaPoints.join(' L ')}" fill="none" stroke="#f72585" stroke-width="2.5" stroke-dasharray="4,3" />

          <!-- Interactive Tracking Cursor -->
          <line id="sched-cursor-line" x1="${padL}" y1="${padT - 5}" x2="${padL}" y2="${padT + plotH + 5}" stroke="#ffffff" stroke-width="2" />
          <circle id="sched-cursor-dot-alpha" cx="${padL}" cy="${padT}" r="4.5" fill="#00f5d4" stroke="#ffffff" stroke-width="1.5" />
          <circle id="sched-cursor-dot-sigma" cx="${padL}" cy="${padT}" r="4.5" fill="#f72585" stroke="#ffffff" stroke-width="1.5" />

          <!-- Axis Labels -->
          <text x="${padL}" y="${padT + plotH + 18}" fill="#64748b" font-size="9" font-family="monospace">t=1000 (Noise)</text>
          <text x="${w - padR - 35}" y="${padT + plotH + 18}" fill="#64748b" font-size="9" font-family="monospace">t=0 (Clean)</text>
          <text x="8" y="${padT + 8}" fill="#00f5d4" font-size="9" font-family="monospace">1.0</text>
          <text x="8" y="${padT + plotH}" fill="#64748b" font-size="9" font-family="monospace">0.0</text>
        </svg>

        <!-- Legend and live readout -->
        <div class="scheduler-meta-row">
          <div class="sched-legend">
            <span class="legend-chip legend-alpha">── Signal (ᾱₜ)</span>
            <span class="legend-chip legend-sigma">--- Noise (σₜ)</span>
          </div>
          <div class="sched-readout" id="sched-readout">
            Step 50: Signal 0.5% • Noise 99.5%
          </div>
        </div>
      </div>
    `;

    // Click on SVG to seek
    const svg = document.getElementById('scheduler-svg');
    if (svg) {
      svg.addEventListener('click', (e) => {
        const rect = svg.getBoundingClientRect();
        const clickX = ((e.clientX - rect.left) / rect.width) * w;
        const normX = (clickX - padL) / plotW;
        if (normX >= 0 && normX <= 1) {
          // Left is step 50, right is step 0
          const step = Math.round(50 - normX * 50);
          this.onSeekStep(step);
        }
      });
    }

    this.updateCursor();
  }

  updateCursor() {
    const line = document.getElementById('sched-cursor-line');
    const dotAlpha = document.getElementById('sched-cursor-dot-alpha');
    const dotSigma = document.getElementById('sched-cursor-dot-sigma');
    const readout = document.getElementById('sched-readout');

    if (!line || !dotAlpha || !dotSigma) return;

    const normX = (this.totalSteps - this.currentStep) / this.totalSteps;
    const x = this.padL + normX * this.plotW;

    const params = getScheduleParams(this.currentStep, this.totalSteps);
    const yAlpha = this.padT + (1.0 - params.alphaBar) * this.plotH;
    const ySigma = this.padT + (1.0 - params.sigma) * this.plotH;

    line.setAttribute('x1', x.toFixed(1));
    line.setAttribute('x2', x.toFixed(1));

    dotAlpha.setAttribute('cx', x.toFixed(1));
    dotAlpha.setAttribute('cy', yAlpha.toFixed(1));

    dotSigma.setAttribute('cx', x.toFixed(1));
    dotSigma.setAttribute('cy', ySigma.toFixed(1));

    if (readout) {
      readout.innerHTML = `
        <strong>Step ${this.currentStep} (t=${params.timestep})</strong>: 
        <span style="color:#00f5d4">Signal ${(params.alphaBar * 100).toFixed(1)}%</span> • 
        <span style="color:#f72585">Noise ${(params.sigma * 100).toFixed(1)}%</span> • 
        SNR: ${params.snrDb.toFixed(1)} dB
      `;
    }
  }
}

/**
 * Classifier-Free Guidance (CFG) Interactive Explorer Component
 * Visualizes the vector math: eps_final = eps_uncond + scale * (eps_cond - eps_uncond)
 * and explains how guidance scale influences creativity vs fidelity.
 */

export class CFGExplorer {
  constructor({ containerId, onScaleChange }) {
    this.container = document.getElementById(containerId);
    this.onScaleChange = onScaleChange || (() => {});
    this.scale = 7.5;

    this.init();
  }

  setScale(scale) {
    this.scale = parseFloat(scale);
    this.update();
  }

  init() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="cfg-card">
        <div class="cfg-header">
          <div class="cfg-title">
            <span class="badge-icon">🎛️</span>
            <span>Classifier-Free Guidance (CFG Scale: <span id="cfg-val-badge">7.5</span>)</span>
          </div>
          <div class="cfg-formula">
            <code>ε̃ = ε<sub>uncond</sub> + <strong>${this.scale}</strong> × (ε<sub>cond</sub> − ε<sub>uncond</sub>)</code>
          </div>
        </div>

        <div class="cfg-controls-row">
          <input type="range" id="cfg-slider" min="1.0" max="20.0" step="0.5" value="${this.scale}" class="styled-range">
          <div class="cfg-range-ticks">
            <span>1.0 (Dreamy/Weak)</span>
            <span class="accent-cyan">7.5 (Optimal Sweet Spot)</span>
            <span>20.0 (Over-saturated)</span>
          </div>
        </div>

        <!-- 2D Vector Math Diagram -->
        <div class="cfg-vector-viewport">
          <svg viewBox="0 0 460 160" class="cfg-vector-svg" id="cfg-vector-svg">
            <!-- Origin point -->
            <circle cx="60" cy="110" r="4" fill="#ffffff" />
            <text x="45" y="130" fill="#64748b" font-size="10" font-family="monospace">Origin</text>

            <!-- Unconditioned Vector (Gray/Dashed) -->
            <line id="vec-uncond" x1="60" y1="110" x2="160" y2="85" stroke="#94a3b8" stroke-width="2" stroke-dasharray="3,3" />
            <circle cx="160" cy="85" r="4" fill="#94a3b8" />
            <text x="135" y="75" fill="#94a3b8" font-size="9" font-family="monospace">ε_uncond (Generic)</text>

            <!-- Conditioned Vector (Cyan) -->
            <line id="vec-cond" x1="60" y1="110" x2="210" y2="55" stroke="#00f5d4" stroke-width="2.5" />
            <circle cx="210" cy="55" r="4" fill="#00f5d4" />
            <text x="215" y="55" fill="#00f5d4" font-size="9" font-family="monospace">ε_cond (Prompt)</text>

            <!-- Difference Vector (Delta, Magenta) -->
            <line id="vec-delta" x1="160" y1="85" x2="210" y2="55" stroke="#f72585" stroke-width="2" />
            
            <!-- Scaled Output Vector (Gold/Orange) -->
            <line id="vec-scaled" x1="160" y1="85" x2="380" y2="20" stroke="#f59e0b" stroke-width="3" stroke-linecap="round" />
            <circle id="dot-scaled" cx="380" cy="20" r="5" fill="#f59e0b" />
            <text id="label-scaled" x="330" y="15" fill="#f59e0b" font-size="10" font-weight="700" font-family="monospace">Final Guided Noise (s=7.5)</text>
          </svg>
        </div>

        <!-- Guidance Behavior Assessment -->
        <div class="cfg-assessment" id="cfg-assessment">
          <span class="badge-status-optimal">🎯 Optimal Balance</span>
          <span>Strong adherence to prompt semantics with natural contrast and realistic photorealism.</span>
        </div>
      </div>
    `;

    const slider = document.getElementById('cfg-slider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        this.scale = parseFloat(e.target.value);
        this.update();
        this.onScaleChange(this.scale);
      });
    }
  }

  update() {
    const badge = document.getElementById('cfg-val-badge');
    if (badge) badge.textContent = this.scale.toFixed(1);

    const vecScaled = document.getElementById('vec-scaled');
    const dotScaled = document.getElementById('dot-scaled');
    const labelScaled = document.getElementById('label-scaled');
    const assessment = document.getElementById('cfg-assessment');

    // Vector calculations:
    // Origin: (60, 110)
    // Uncond endpoint: (160, 85) -> dx = 100, dy = -25
    // Cond endpoint: (210, 55) -> dx = 150, dy = -55
    // Delta = (50, -30)
    // Scaled end = Uncond + (scale / 7.5) * (150, -65)
    const factor = this.scale / 7.5;
    const endX = Math.min(440, Math.max(170, 160 + 160 * factor * 0.7));
    const endY = Math.max(10, 85 - 55 * factor);

    if (vecScaled && dotScaled) {
      vecScaled.setAttribute('x2', endX.toFixed(1));
      vecScaled.setAttribute('y2', endY.toFixed(1));
      dotScaled.setAttribute('cx', endX.toFixed(1));
      dotScaled.setAttribute('cy', endY.toFixed(1));
    }

    if (labelScaled) {
      labelScaled.setAttribute('x', Math.min(320, endX - 60).toFixed(1));
      labelScaled.setAttribute('y', Math.max(15, endY - 8).toFixed(1));
      labelScaled.textContent = `Final Guided Vector (s=${this.scale.toFixed(1)})`;
    }

    if (assessment) {
      if (this.scale <= 3.0) {
        assessment.innerHTML = `
          <span class="badge-status-weak">⚠️ Weak Guidance (s=${this.scale})</span>
          <span>Low prompt adherence. Image may ignore prompt keywords and wander into generic compositions.</span>
        `;
      } else if (this.scale <= 9.0) {
        assessment.innerHTML = `
          <span class="badge-status-optimal">🎯 Optimal Sweet Spot (s=${this.scale})</span>
          <span>Crisp semantic alignment, sharp contours, rich natural dynamic range, and minimal artifacts.</span>
        `;
      } else {
        assessment.innerHTML = `
          <span class="badge-status-high">🔥 Over-Saturated Guidance (s=${this.scale})</span>
          <span>High guidance pushes vectors outside latent manifold, causing burnt colors, unnatural contrast, and noisy halos.</span>
        `;
      }
    }
  }
}

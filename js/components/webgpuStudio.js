/**
 * WebGPU Stable Diffusion Studio UI Controller
 * Manages the user interface for custom prompt generation, live step rendering,
 * parameter sliders, token attention inspection, and the generation history gallery.
 */

import { WebGPUDiffusionEngine } from './webgpuEngine.js';
import { computeAttentionHeatmap } from '../utils/math.js';
import { COLORMAPS, getRGB } from '../utils/colormaps.js';

export class WebGPUStudio {
  constructor({ containerId, onInspectInExplainer }) {
    this.container = document.getElementById(containerId);
    this.onInspectInExplainer = onInspectInExplainer || (() => {});
    this.engine = new WebGPUDiffusionEngine({
      onStatusChange: this.handleEngineStatus.bind(this)
    });

    this.history = [];
    this.activeGeneration = null;
    this.selectedToken = null;
    this.tokenData = null;
    this.colormap = COLORMAPS.TURBO;

    // Cache elements once DOM is ready
    this.dom = {};
  }

  async init() {
    this.render();
    this.cacheDOMElements();
    this.bindEvents();

    // Initialize WebGPU hardware check
    await this.engine.init();
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="webgpu-studio-wrapper">
        <!-- Top GPU Hardware Status & Capability Banner -->
        <div class="webgpu-status-banner" id="webgpu-hardware-banner">
          <div class="gpu-status-left">
            <span class="gpu-status-dot" id="gpu-status-dot"></span>
            <span class="gpu-status-text" id="gpu-status-text">Detecting WebGPU Hardware...</span>
          </div>
          <div class="gpu-status-telemetry" id="gpu-status-telemetry">
            <span class="badge-metric" id="badge-gpu-vendor">Checking GPU...</span>
            <span class="badge-metric" id="badge-gpu-buffer">Max Buffer: -- MB</span>
            <span class="badge-metric" id="badge-gpu-perf">Ready</span>
          </div>
        </div>

        <div class="webgpu-studio-grid">
          <!-- LEFT COLUMN: PROMPT INPUT & SETTINGS CONTROLS -->
          <div class="webgpu-controls-column">
            <!-- Custom Prompt Card -->
            <div class="studio-card">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>✨</span> Custom Prompt Studio
                </span>
                <span class="card-badge" id="prompt-char-count">0 chars</span>
              </div>

              <!-- Main Prompt Input Textarea -->
              <div class="prompt-input-group">
                <textarea 
                  class="styled-textarea" 
                  id="studio-prompt-input" 
                  rows="3" 
                  placeholder="Enter any custom prompt, e.g. 'a cute fluffy red panda wearing an astronaut helmet, cinematic lighting'..."
                >a cyberpunk samurai cat with glowing neon katana, rain soaked street at night, highly detailed</textarea>
              </div>

              <!-- Quick Prompt Inspiration Chips -->
              <div class="quick-prompts-bar">
                <span class="quick-label">Inspiration:</span>
                <button class="quick-chip" data-prompt="a cute fluffy red panda wearing an astronaut helmet, cinematic lighting" type="button">🐼 Astro Panda</button>
                <button class="quick-chip" data-prompt="cyberpunk city street at night in rain with glowing neon signs" type="button">🌆 Cyber Neon</button>
                <button class="quick-chip" data-prompt="a majestic fantasy castle perched on a floating island with waterfalls" type="button">🏰 Castle Island</button>
                <button class="quick-chip" data-prompt="a serene mountain lake at sunrise, vibrant watercolor painting" type="button">🎨 Watercolor Lake</button>
                <button class="quick-chip" data-prompt="steampunk mechanical dragon perched on clocktower, brass and copper" type="button">🐉 Steampunk Dragon</button>
              </div>

              <!-- Negative Prompt (Collapsible) -->
              <details class="negative-prompt-details">
                <summary class="negative-prompt-summary">
                  <span>🚫 Negative Prompt (Optional)</span>
                  <span style="font-size:0.7rem; color:var(--text-muted);">(filter unwanted traits)</span>
                </summary>
                <div style="padding-top:8px;">
                  <textarea 
                    class="styled-textarea styled-textarea-sm" 
                    id="studio-negative-prompt" 
                    rows="2" 
                    placeholder="blurry, bad anatomy, low quality, artifacts, distorted, noisy"
                  >blurry, low quality, distorted, artifacts, noisy, bad composition</textarea>
                </div>
              </details>
            </div>

            <!-- Diffusion Hyperparameters & Settings Card -->
            <div class="studio-card">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>⚙️</span> Generation Parameters
                </span>
                <span class="card-badge" id="steps-badge">15 Steps</span>
              </div>

              <div class="settings-grid">
                <!-- Inference Steps Slider -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="slider-steps">Inference Steps:</label>
                    <span class="setting-val-display" id="val-steps">15</span>
                  </div>
                  <input type="range" id="slider-steps" min="1" max="25" value="15" step="1" class="styled-range">
                  <div class="setting-hint">1-4 (LCM fast) • 12-15 (Balanced) • 20-25 (High fidelity)</div>
                </div>

                <!-- Guidance Scale (CFG) Slider -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="slider-cfg">Guidance Scale (CFG):</label>
                    <span class="setting-val-display" id="val-cfg">7.5</span>
                  </div>
                  <input type="range" id="slider-cfg" min="1.0" max="15.0" value="7.5" step="0.5" class="styled-range">
                  <div class="setting-hint">1.0 (Creative/Free) • 7.5 (Optimal) • 15.0 (Strict/Sharp)</div>
                </div>

                <!-- Seed Control -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="input-seed">Random Seed:</label>
                    <button class="btn-dice" id="btn-random-seed" type="button" title="Randomize Seed">🎲 Randomize</button>
                  </div>
                  <div style="display:flex; gap:6px;">
                    <input type="number" id="input-seed" class="styled-number-input" value="4242" placeholder="Random">
                  </div>
                </div>

                <!-- Output Resolution -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="select-resolution">Resolution:</label>
                    <span class="setting-val-display" id="val-res">512×512</span>
                  </div>
                  <select id="select-resolution" class="styled-select" style="width:100%;">
                    <option value="256">256×256 (Ultra-Fast Preview)</option>
                    <option value="512" selected>512×512 (Standard Stable Diffusion)</option>
                  </select>
                </div>
              </div>

              <!-- Main Generation Trigger Button -->
              <div class="generate-actions-row">
                <button class="btn-generate-webgpu" id="btn-generate" type="button">
                  <span class="btn-icon">⚡</span>
                  <span class="btn-label" id="btn-generate-label">Generate on WebGPU</span>
                </button>
                <button class="btn-cancel hidden" id="btn-cancel" type="button" title="Cancel generation">⏹️ Stop</button>
              </div>
            </div>
          </div>

          <!-- RIGHT COLUMN: LIVE DIFFUSION VIEWPORT & CANVASES -->
          <div class="webgpu-viewport-column">
            <div class="studio-card viewport-card">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>🎨</span> Live Diffusion Synthesis
                </span>
                <div class="viewport-header-badges">
                  <span class="badge-metric badge-cyan" id="live-step-status">Ready</span>
                  <span class="badge-metric" id="live-latency-status">-- ms/step</span>
                </div>
              </div>

              <!-- Real-Time Progress Bar -->
              <div class="webgpu-progress-track">
                <div class="webgpu-progress-fill" id="webgpu-progress-bar" style="width: 0%;"></div>
              </div>

              <!-- Dual Live Visual Viewport: Decoded Canvas & 64x64 Latent Overlay -->
              <div class="live-viewport-stage">
                <!-- Main Decoded RGB Canvas (512x512) -->
                <canvas class="live-canvas-main" id="studio-main-canvas" width="512" height="512"></canvas>
                
                <!-- Cross-Attention Spatial Overlay Canvas -->
                <canvas class="live-canvas-attention" id="studio-attention-canvas" width="512" height="512"></canvas>

                <!-- Floating Latent Space Preview Window (64x64x4) -->
                <div class="floating-latent-window" id="floating-latent-window">
                  <div class="floating-latent-header">
                    <span>🧬 64×64 Latent</span>
                    <span id="floating-latent-step">t=1000</span>
                  </div>
                  <canvas class="floating-latent-canvas" id="studio-latent-canvas" width="64" height="64"></canvas>
                </div>
              </div>

              <!-- Dynamic Custom Prompt Tokenizer & Attention Chips -->
              <div class="custom-tokens-section">
                <div class="custom-tokens-header">
                  <span style="font-size:0.75rem; font-weight:700; color:var(--text-secondary);">PROMPT ATTENTION CHIPS (Click to overlay spatial heatmap):</span>
                </div>
                <div class="custom-tokens-list" id="studio-tokens-list">
                  <span style="font-size:0.75rem; color:var(--text-muted);">Enter a prompt and click Generate to see interactive token attention.</span>
                </div>
              </div>

              <!-- Generation Action Tools (Download, Copy, Send to Explainer) -->
              <div class="viewport-actions-bar">
                <button class="btn-tool" id="btn-download-png" type="button" disabled>
                  <span>📥</span> Download PNG
                </button>
                <button class="btn-tool" id="btn-copy-settings" type="button" disabled>
                  <span>📋</span> Copy Seed & Prompt
                </button>
                <button class="btn-tool" id="btn-send-explainer" type="button">
                  <span>🔄</span> Send to Explainer
                </button>
              </div>
            </div>

            <!-- Generation History Gallery -->
            <div class="studio-card history-card">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>🖼️</span> Session Generation History
                </span>
                <span class="card-badge" id="history-count-badge">0 items</span>
              </div>
              <div class="history-gallery-grid" id="history-gallery-grid">
                <div class="history-empty-msg">No images generated yet this session. Click Generate to create your first image!</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  cacheDOMElements() {
    this.dom = {
      promptInput: document.getElementById('studio-prompt-input'),
      negativePromptInput: document.getElementById('studio-negative-prompt'),
      charCount: document.getElementById('prompt-char-count'),
      sliderSteps: document.getElementById('slider-steps'),
      valSteps: document.getElementById('val-steps'),
      stepsBadge: document.getElementById('steps-badge'),
      sliderCfg: document.getElementById('slider-cfg'),
      valCfg: document.getElementById('val-cfg'),
      inputSeed: document.getElementById('input-seed'),
      btnRandomSeed: document.getElementById('btn-random-seed'),
      selectResolution: document.getElementById('select-resolution'),
      valRes: document.getElementById('val-res'),
      btnGenerate: document.getElementById('btn-generate'),
      btnGenerateLabel: document.getElementById('btn-generate-label'),
      btnCancel: document.getElementById('btn-cancel'),
      mainCanvas: document.getElementById('studio-main-canvas'),
      attentionCanvas: document.getElementById('studio-attention-canvas'),
      latentCanvas: document.getElementById('studio-latent-canvas'),
      floatingLatentStep: document.getElementById('floating-latent-step'),
      progressBar: document.getElementById('webgpu-progress-bar'),
      liveStepStatus: document.getElementById('live-step-status'),
      liveLatencyStatus: document.getElementById('live-latency-status'),
      tokensList: document.getElementById('studio-tokens-list'),
      btnDownloadPng: document.getElementById('btn-download-png'),
      btnCopySettings: document.getElementById('btn-copy-settings'),
      btnSendExplainer: document.getElementById('btn-send-explainer'),
      historyGrid: document.getElementById('history-gallery-grid'),
      historyCountBadge: document.getElementById('history-count-badge'),
      gpuStatusDot: document.getElementById('gpu-status-dot'),
      gpuStatusText: document.getElementById('gpu-status-text'),
      badgeGpuVendor: document.getElementById('badge-gpu-vendor'),
      badgeGpuBuffer: document.getElementById('badge-gpu-buffer'),
      badgeGpuPerf: document.getElementById('badge-gpu-perf')
    };
  }

  bindEvents() {
    // Prompt character counter
    if (this.dom.promptInput) {
      this.dom.promptInput.addEventListener('input', () => {
        const len = this.dom.promptInput.value.length;
        if (this.dom.charCount) this.dom.charCount.textContent = `${len} chars`;
      });
    }

    // Quick Inspiration Prompt Chips
    this.container.querySelectorAll('.quick-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const p = chip.dataset.prompt;
        if (this.dom.promptInput) {
          this.dom.promptInput.value = p;
          if (this.dom.charCount) this.dom.charCount.textContent = `${p.length} chars`;
        }
      });
    });

    // Steps slider
    if (this.dom.sliderSteps) {
      this.dom.sliderSteps.addEventListener('input', (e) => {
        const val = e.target.value;
        if (this.dom.valSteps) this.dom.valSteps.textContent = val;
        if (this.dom.stepsBadge) this.dom.stepsBadge.textContent = `${val} Steps`;
      });
    }

    // CFG slider
    if (this.dom.sliderCfg) {
      this.dom.sliderCfg.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(1);
        if (this.dom.valCfg) this.dom.valCfg.textContent = val;
      });
    }

    // Resolution selector
    if (this.dom.selectResolution) {
      this.dom.selectResolution.addEventListener('change', (e) => {
        const r = e.target.value;
        if (this.dom.valRes) this.dom.valRes.textContent = `${r}×${r}`;
        if (this.dom.mainCanvas) {
          this.dom.mainCanvas.width = parseInt(r, 10);
          this.dom.mainCanvas.height = parseInt(r, 10);
        }
        if (this.dom.attentionCanvas) {
          this.dom.attentionCanvas.width = parseInt(r, 10);
          this.dom.attentionCanvas.height = parseInt(r, 10);
        }
      });
    }

    // Randomize seed button
    if (this.dom.btnRandomSeed) {
      this.dom.btnRandomSeed.addEventListener('click', () => {
        const newSeed = Math.floor(Math.random() * 900000) + 1000;
        if (this.dom.inputSeed) this.dom.inputSeed.value = newSeed;
      });
    }

    // Generate Button
    if (this.dom.btnGenerate) {
      this.dom.btnGenerate.addEventListener('click', () => {
        this.startGeneration();
      });
    }

    // Cancel Button
    if (this.dom.btnCancel) {
      this.dom.btnCancel.addEventListener('click', () => {
        this.engine.cancel();
      });
    }

    // Download PNG Button
    if (this.dom.btnDownloadPng) {
      this.dom.btnDownloadPng.addEventListener('click', () => {
        this.downloadCurrentPng();
      });
    }

    // Copy Prompt & Seed
    if (this.dom.btnCopySettings) {
      this.dom.btnCopySettings.addEventListener('click', () => {
        if (!this.activeGeneration) return;
        const text = `Prompt: "${this.activeGeneration.prompt}"\nSeed: ${this.activeGeneration.seed}\nSteps: ${this.activeGeneration.steps}\nCFG: ${this.activeGeneration.guidanceScale}`;
        navigator.clipboard.writeText(text).then(() => {
          this.dom.btnCopySettings.innerHTML = '<span>✅</span> Copied!';
          setTimeout(() => {
            this.dom.btnCopySettings.innerHTML = '<span>📋</span> Copy Seed & Prompt';
          }, 2000);
        });
      });
    }

    // Send to Explainer Button
    if (this.dom.btnSendExplainer) {
      this.dom.btnSendExplainer.addEventListener('click', () => {
        this.onInspectInExplainer();
      });
    }
  }

  handleEngineStatus({ status, message, backend, deviceInfo }) {
    if (this.dom.gpuStatusText) this.dom.gpuStatusText.textContent = message;

    if (this.dom.gpuStatusDot) {
      if (backend === 'webgpu') {
        this.dom.gpuStatusDot.className = 'gpu-status-dot dot-active';
      } else {
        this.dom.gpuStatusDot.className = 'gpu-status-dot dot-warning';
      }
    }

    if (deviceInfo) {
      if (this.dom.badgeGpuVendor) this.dom.badgeGpuVendor.textContent = deviceInfo.vendor || 'GPU Active';
      if (this.dom.badgeGpuBuffer) this.dom.badgeGpuBuffer.textContent = `Max Buffer: ${deviceInfo.maxBufferSizeMB} MB`;
      if (this.dom.badgeGpuPerf) this.dom.badgeGpuPerf.textContent = backend === 'webgpu' ? '⚡ WebGPU Accelerated' : 'Wasm CPU Mode';
    }
  }

  async startGeneration() {
    const prompt = this.dom.promptInput ? this.dom.promptInput.value.trim() : '';
    const negativePrompt = this.dom.negativePromptInput ? this.dom.negativePromptInput.value.trim() : '';
    const steps = parseInt(this.dom.sliderSteps.value, 10) || 15;
    const guidanceScale = parseFloat(this.dom.sliderCfg.value) || 7.5;
    const seed = this.dom.inputSeed.value !== '' ? parseInt(this.dom.inputSeed.value, 10) : Math.floor(Math.random() * 1000000);
    const size = parseInt(this.dom.selectResolution.value, 10) || 512;

    if (!prompt) {
      alert('Please enter a prompt to generate!');
      return;
    }

    // Update UI for generation state
    this.dom.btnGenerate.disabled = true;
    this.dom.btnGenerateLabel.textContent = 'Generating on WebGPU...';
    this.dom.btnCancel.classList.remove('hidden');
    if (this.dom.progressBar) this.dom.progressBar.style.width = '0%';
    if (this.dom.liveStepStatus) this.dom.liveStepStatus.textContent = 'Initializing Latents...';

    const mainCtx = this.dom.mainCanvas.getContext('2d');
    const latentCtx = this.dom.latentCanvas.getContext('2d');
    const attCtx = this.dom.attentionCanvas.getContext('2d');
    attCtx.clearRect(0, 0, size, size);

    try {
      await this.engine.generate({
        prompt,
        negativePrompt,
        steps,
        guidanceScale,
        seed,
        width: size,
        height: size,
        onStep: (stepInfo) => {
          const pct = Math.round((stepInfo.currentStep / stepInfo.totalSteps) * 100);
          if (this.dom.progressBar) this.dom.progressBar.style.width = `${pct}%`;

          if (this.dom.liveStepStatus) {
            if (stepInfo.currentStep === 0) {
              this.dom.liveStepStatus.textContent = `Step 0/${stepInfo.totalSteps} (Initial Noise t=1000)`;
            } else {
              this.dom.liveStepStatus.textContent = `Step ${stepInfo.currentStep}/${stepInfo.totalSteps} (t=${stepInfo.timestep})`;
            }
          }
          if (this.dom.liveLatencyStatus) {
            this.dom.liveLatencyStatus.textContent = `${Math.round(stepInfo.stepLatencyMs)}ms/step`;
          }
          if (this.dom.floatingLatentStep) {
            this.dom.floatingLatentStep.textContent = `t=${stepInfo.timestep}`;
          }

          // Draw current step decoded frame onto main canvas
          if (stepInfo.frameCanvas) {
            mainCtx.drawImage(stepInfo.frameCanvas, 0, 0, size, size);
          }

          // Draw raw 64x64 latent PCA directly onto floating preview window
          if (this.engine.latentCanvas) {
            latentCtx.drawImage(this.engine.latentCanvas, 0, 0, 64, 64);
          } else if (stepInfo.frameCanvas) {
            latentCtx.drawImage(stepInfo.frameCanvas, 0, 0, 64, 64);
          }

          // Initialize token attention chips as soon as prompt is tokenized
          if (!this.tokenData && stepInfo.tokens) {
            this.tokenData = { tokens: stepInfo.tokens, hotspots: stepInfo.hotspots };
            this.renderTokenChips(stepInfo.tokens, stepInfo.hotspots);
          }

          // Render cross attention if active
          if (this.selectedToken && this.tokenData?.hotspots[this.selectedToken]) {
            this.renderAttentionOverlay(this.tokenData.hotspots[this.selectedToken], size);
          }
        },
        onComplete: (result) => {
          this.activeGeneration = result;
          if (this.dom.btnDownloadPng) this.dom.btnDownloadPng.disabled = false;
          if (this.dom.btnCopySettings) this.dom.btnCopySettings.disabled = false;

          // Add to History Gallery
          this.addToHistory(result);

          if (this.dom.liveStepStatus) {
            const timeSec = (result.totalTimeMs / 1000).toFixed(1);
            const backendName = result.backend === 'webgpu' ? '⚡ WebGPU' : '💻 CPU SIMD';
            this.dom.liveStepStatus.textContent = `Completed (${backendName}) in ${timeSec}s`;
          }
        }
      });
    } catch (err) {
      console.error('WebGPU Generation error:', err);
      if (this.dom.liveStepStatus) this.dom.liveStepStatus.textContent = 'Error during generation';
    } finally {
      this.dom.btnGenerate.disabled = false;
      this.dom.btnGenerateLabel.textContent = 'Generate on WebGPU';
      this.dom.btnCancel.classList.add('hidden');
    }
  }

  renderTokenChips(tokens, hotspots) {
    if (!this.dom.tokensList || !tokens) return;

    this.dom.tokensList.innerHTML = tokens.map((t, idx) => {
      const hasSpot = !!hotspots[t];
      const isSelected = this.selectedToken === t;
      return `
        <button 
          class="studio-token-chip ${isSelected ? 'active' : ''} ${hasSpot ? 'has-focus' : ''}" 
          data-token="${t}" 
          type="button"
          title="${hasSpot ? 'Strong Semantic Cross-Attention Focus' : 'Context Token'}"
        >
          <span>${t}</span>
          ${hasSpot ? '<span class="studio-token-dot"></span>' : ''}
        </button>
      `;
    }).join('');

    this.dom.tokensList.querySelectorAll('.studio-token-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const token = chip.dataset.token;
        if (this.selectedToken === token) {
          this.selectedToken = null;
          chip.classList.remove('active');
          const size = this.dom.attentionCanvas.width;
          const attCtx = this.dom.attentionCanvas.getContext('2d');
          attCtx.clearRect(0, 0, size, size);
        } else {
          this.selectedToken = token;
          this.dom.tokensList.querySelectorAll('.studio-token-chip').forEach(c => c.classList.remove('active'));
          chip.classList.add('active');

          if (this.tokenData && this.tokenData.hotspots[token]) {
            const size = this.dom.attentionCanvas.width;
            this.renderAttentionOverlay(this.tokenData.hotspots[token], size);
          }
        }
      });
    });
  }

  renderAttentionOverlay(hotspot, size) {
    const canvas = this.dom.attentionCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, size, size);

    if (!hotspot) return;

    const map = computeAttentionHeatmap(hotspot, 64, 64);
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 64;
    offCanvas.height = 64;
    const offCtx = offCanvas.getContext('2d');
    const imgData = offCtx.createImageData(64, 64);

    for (let i = 0; i < map.length; i++) {
      const val = map[i];
      const [r, g, b] = getRGB(val, this.colormap);
      const px = i * 4;
      imgData.data[px] = r;
      imgData.data[px + 1] = g;
      imgData.data[px + 2] = b;
      imgData.data[px + 3] = Math.round(Math.pow(val, 0.75) * 180);
    }

    offCtx.putImageData(imgData, 0, 0);

    ctx.save();
    ctx.globalAlpha = 0.65;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offCanvas, 0, 0, size, size);
    ctx.restore();
  }

  addToHistory(result) {
    const dataUrl = result.canvas.toDataURL('image/png');
    const item = {
      id: Date.now(),
      prompt: result.prompt,
      seed: result.seed,
      steps: result.steps,
      guidanceScale: result.guidanceScale,
      dataUrl
    };

    this.history.unshift(item);
    this.renderHistory();
  }

  renderHistory() {
    if (!this.dom.historyGrid) return;

    if (this.dom.historyCountBadge) {
      this.dom.historyCountBadge.textContent = `${this.history.length} items`;
    }

    if (this.history.length === 0) {
      this.dom.historyGrid.innerHTML = '<div class="history-empty-msg">No images generated yet this session. Click Generate to create your first image!</div>';
      return;
    }

    this.dom.historyGrid.innerHTML = this.history.map(item => `
      <div class="history-item" data-id="${item.id}" title="${escapeHtml(item.prompt)} (Seed: ${item.seed})">
        <img class="history-thumb" src="${item.dataUrl}" alt="Generated image thumbnail">
        <div class="history-item-overlay">
          <span class="history-seed-badge">#${item.seed}</span>
        </div>
      </div>
    `).join('');

    this.dom.historyGrid.querySelectorAll('.history-item').forEach(elem => {
      elem.addEventListener('click', () => {
        const id = parseInt(elem.dataset.id, 10);
        const match = this.history.find(h => h.id === id);
        if (match) {
          // Restore prompt and image
          if (this.dom.promptInput) this.dom.promptInput.value = match.prompt;
          if (this.dom.inputSeed) this.dom.inputSeed.value = match.seed;
          if (this.dom.sliderSteps) {
            this.dom.sliderSteps.value = match.steps;
            if (this.dom.valSteps) this.dom.valSteps.textContent = match.steps;
          }

          const img = new Image();
          img.onload = () => {
            const ctx = this.dom.mainCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, this.dom.mainCanvas.width, this.dom.mainCanvas.height);
          };
          img.src = match.dataUrl;
        }
      });
    });
  }

  downloadCurrentPng() {
    if (!this.dom.mainCanvas) return;
    const link = document.createElement('a');
    link.download = `stable-diffusion-webgpu-seed-${this.activeGeneration?.seed || 'gen'}.png`;
    link.href = this.dom.mainCanvas.toDataURL('image/png');
    link.click();
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

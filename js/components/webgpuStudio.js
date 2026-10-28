/**
 * WebGPU Stable Diffusion Studio UI Controller
 * Manages the user interface for:
 * 1. Real In-Browser WebGPU Neural Network (Stability AI SD-Turbo ONNX)
 * 2. Local / Cloud Server API Connection (Automatic1111, ComfyUI, Hugging Face)
 * 3. Interactive Latent Diffusion Math Simulator (50ms Instant Exploration)
 */

import { WebGPUNeuralEngine } from './webgpuNeuralEngine.js';
import { ServerBackendEngine } from './serverBackendEngine.js';
import { WebGPUDiffusionEngine } from './webgpuEngine.js';
import { computeAttentionHeatmap } from '../utils/math.js';
import { COLORMAPS, getRGB } from '../utils/colormaps.js';

export class WebGPUStudio {
  constructor({ containerId, onInspectInExplainer }) {
    this.container = document.getElementById(containerId);
    this.onInspectInExplainer = onInspectInExplainer || (() => {});

    // Active Engine Mode: 'neural' | 'backend' | 'simulator'
    this.engineMode = 'neural';

    // 1. Real In-Browser WebGPU Neural Engine (SD-Turbo ONNX)
    this.neuralEngine = new WebGPUNeuralEngine({
      onStatusChange: this.handleNeuralStatus.bind(this),
      onProgress: this.handleDownloadProgress.bind(this)
    });

    // 2. Direct Server / Cloud API Engine
    this.serverEngine = new ServerBackendEngine();

    // 3. Mathematical Latent Simulator Engine
    this.simEngine = new WebGPUDiffusionEngine({
      onStatusChange: this.handleSimStatus.bind(this)
    });

    this.history = [];
    this.activeGeneration = null;
    this.selectedToken = null;
    this.tokenData = null;
    this.colormap = COLORMAPS.TURBO;

    this.dom = {};
  }

  async init() {
    this.render();
    this.cacheDOMElements();
    this.bindEvents();

    // Initialize both engines in parallel
    await Promise.all([
      this.neuralEngine.init(),
      this.simEngine.init()
    ]);
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
          <!-- LEFT COLUMN: PROMPT INPUT, ENGINE SELECTOR & SETTINGS -->
          <div class="webgpu-controls-column">
            
            <!-- Generation Engine Mode Selector Tabs -->
            <div class="engine-mode-card">
              <div class="engine-mode-tabs">
                <button class="engine-mode-tab active" data-mode="neural" id="tab-mode-neural" type="button" title="Genuine Stability AI SD-Turbo model running via WebGPU">
                  <span>🧠</span> Real WebGPU Neural Model
                </button>
                <button class="engine-mode-tab" data-mode="backend" id="tab-mode-backend" type="button" title="Connect to local Automatic1111, ComfyUI, or free HuggingFace API">
                  <span>🔗</span> Local / Cloud API
                </button>
                <button class="engine-mode-tab" data-mode="simulator" id="tab-mode-simulator" type="button" title="Instant 50ms mathematical simulator for learning diffusion curves">
                  <span>📐</span> Math Simulator
                </button>
              </div>
            </div>

            <!-- 1. Real WebGPU Neural Network Model Weights Manager Card -->
            <div class="studio-card model-weights-card" id="card-neural-manager">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>📦</span> Stability AI SD-Turbo (ONNX FP16)
                </span>
                <span class="card-badge" id="badge-model-status">Checking Cache...</span>
              </div>
              <p class="model-desc">
                Executes the actual 860M-parameter U-Net, CLIP text encoder, and VAE decoder directly on your GPU using ONNX Runtime Web. Model files (~2.4 GB) are cached locally in your browser's IndexedDB.
              </p>
              
              <!-- Download & Initialize Action Buttons -->
              <div class="model-action-row" id="model-action-row">
                <button class="btn-download-weights" id="btn-load-weights" type="button">
                  <span id="btn-load-weights-icon">📥</span>
                  <span id="btn-load-weights-text">Download & Load Weights (~2.4 GB)</span>
                </button>
                <button class="btn-clear-cache hidden" id="btn-clear-cache" type="button" title="Delete cached weights from IndexedDB">
                  <span>🗑️</span> Clear Cache
                </button>
              </div>

              <!-- Real-time Download Progress Bar & Speed Stats -->
              <div class="model-download-progress-container hidden" id="model-progress-container">
                <div class="download-labels-row">
                  <span class="download-task-name" id="download-task-name">Downloading...</span>
                  <span class="download-pct" id="download-pct">0%</span>
                </div>
                <div class="download-track">
                  <div class="download-fill" id="download-progress-fill" style="width: 0%;"></div>
                </div>
                <div class="download-stats-row">
                  <span id="download-bytes-info">0 MB / 2,450 MB</span>
                  <span id="download-speed-info">0 MB/s • ETA: --s</span>
                </div>
              </div>
            </div>

            <!-- 2. Local / Cloud Server API Configuration Card -->
            <div class="studio-card hidden" id="card-backend-settings">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>🔗</span> Backend Server Configuration
                </span>
                <span class="card-badge">Zero Download</span>
              </div>
              <p class="model-desc">
                Connect to your local GPU workstation or Hugging Face serverless inference to generate images instantly without downloading 2.4 GB of browser weights.
              </p>
              <div class="settings-grid">
                <div class="setting-item">
                  <label for="select-backend-type">Server Backend:</label>
                  <select id="select-backend-type" class="styled-select" style="width:100%;">
                    <option value="auto1111" selected>Automatic1111 / SD.Next (127.0.0.1:7860)</option>
                    <option value="hf-api">Hugging Face Serverless Inference API</option>
                  </select>
                </div>
                <div class="setting-item" id="group-backend-url">
                  <label for="input-backend-url">Endpoint URL:</label>
                  <input type="text" id="input-backend-url" class="styled-text-input" value="http://127.0.0.1:7860">
                </div>
                <div class="setting-item hidden" id="group-hf-token">
                  <label for="input-hf-token">Hugging Face API Token:</label>
                  <input type="password" id="input-hf-token" class="styled-text-input" placeholder="hf_xxxxxxxxxxxxxxxxxxxx">
                </div>
              </div>
            </div>

            <!-- 3. Math Simulator Info Card -->
            <div class="studio-card hidden" id="card-simulator-info" style="border-left: 3px solid var(--accent-magenta);">
              <div class="studio-card-header">
                <span class="studio-card-title">
                  <span>📐</span> Latent Diffusion Math Simulator
                </span>
                <span class="card-badge" style="background:rgba(247,37,133,0.2); color:var(--accent-magenta);">Educational Preview</span>
              </div>
              <p class="model-desc">
                Interactive mathematical exploration of DDIM reverse diffusion equations, frequency cutoffs, and 4-channel PCA latent projections. Renders instantly in 50ms with 0 MB download.
              </p>
            </div>

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
                <span class="card-badge" id="steps-badge">1-4 Steps</span>
              </div>

              <div class="settings-grid">
                <!-- Inference Steps Slider -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="slider-steps">Inference Steps:</label>
                    <span class="setting-val-display" id="val-steps">1</span>
                  </div>
                  <input type="range" id="slider-steps" min="1" max="4" value="1" step="1" class="styled-range">
                  <div class="setting-hint" id="steps-hint">1-4 steps (Distilled SD-Turbo optimal)</div>
                </div>

                <!-- Guidance Scale (CFG) Slider -->
                <div class="setting-item">
                  <div class="setting-label-row">
                    <label for="slider-cfg">Guidance Scale (CFG):</label>
                    <span class="setting-val-display" id="val-cfg">0.0</span>
                  </div>
                  <input type="range" id="slider-cfg" min="0.0" max="15.0" value="0.0" step="0.5" class="styled-range">
                  <div class="setting-hint" id="cfg-hint">0.0 (Optimal for distilled SD-Turbo)</div>
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
                    <option value="512" selected>512×512 (Standard Stable Diffusion)</option>
                    <option value="256">256×256 (Fast Preview)</option>
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
                  <span class="badge-metric" id="live-latency-status">-- ms</span>
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
                  <span style="font-size:0.75rem; font-weight:700; color:var(--text-secondary);">PROMPT ATTENTION CHIPS (Click to inspect spatial attention):</span>
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
      // Mode tabs & cards
      tabModeNeural: document.getElementById('tab-mode-neural'),
      tabModeBackend: document.getElementById('tab-mode-backend'),
      tabModeSimulator: document.getElementById('tab-mode-simulator'),
      cardNeuralManager: document.getElementById('card-neural-manager'),
      cardBackendSettings: document.getElementById('card-backend-settings'),
      cardSimulatorInfo: document.getElementById('card-simulator-info'),

      // Neural weight manager
      badgeModelStatus: document.getElementById('badge-model-status'),
      btnLoadWeights: document.getElementById('btn-load-weights'),
      btnLoadWeightsIcon: document.getElementById('btn-load-weights-icon'),
      btnLoadWeightsText: document.getElementById('btn-load-weights-text'),
      btnClearCache: document.getElementById('btn-clear-cache'),
      modelProgressContainer: document.getElementById('model-progress-container'),
      downloadTaskName: document.getElementById('download-task-name'),
      downloadPct: document.getElementById('download-pct'),
      downloadProgressFill: document.getElementById('download-progress-fill'),
      downloadBytesInfo: document.getElementById('download-bytes-info'),
      downloadSpeedInfo: document.getElementById('download-speed-info'),

      // Backend API inputs
      selectBackendType: document.getElementById('select-backend-type'),
      inputBackendUrl: document.getElementById('input-backend-url'),
      groupBackendUrl: document.getElementById('group-backend-url'),
      inputHfToken: document.getElementById('input-hf-token'),
      groupHfToken: document.getElementById('group-hf-token'),

      // Prompt & Generation Controls
      promptInput: document.getElementById('studio-prompt-input'),
      negativePromptInput: document.getElementById('studio-negative-prompt'),
      charCount: document.getElementById('prompt-char-count'),
      sliderSteps: document.getElementById('slider-steps'),
      valSteps: document.getElementById('val-steps'),
      stepsBadge: document.getElementById('steps-badge'),
      stepsHint: document.getElementById('steps-hint'),
      sliderCfg: document.getElementById('slider-cfg'),
      valCfg: document.getElementById('val-cfg'),
      cfgHint: document.getElementById('cfg-hint'),
      inputSeed: document.getElementById('input-seed'),
      btnRandomSeed: document.getElementById('btn-random-seed'),
      selectResolution: document.getElementById('select-resolution'),
      valRes: document.getElementById('val-res'),
      btnGenerate: document.getElementById('btn-generate'),
      btnGenerateLabel: document.getElementById('btn-generate-label'),
      btnCancel: document.getElementById('btn-cancel'),

      // Canvas & Viewport
      mainCanvas: document.getElementById('studio-main-canvas'),
      attentionCanvas: document.getElementById('studio-attention-canvas'),
      latentCanvas: document.getElementById('studio-latent-canvas'),
      floatingLatentStep: document.getElementById('floating-latent-step'),
      progressBar: document.getElementById('webgpu-progress-bar'),
      liveStepStatus: document.getElementById('live-step-status'),
      liveLatencyStatus: document.getElementById('live-latency-status'),
      tokensList: document.getElementById('studio-tokens-list'),

      // Actions & Gallery
      btnDownloadPng: document.getElementById('btn-download-png'),
      btnCopySettings: document.getElementById('btn-copy-settings'),
      btnSendExplainer: document.getElementById('btn-send-explainer'),
      historyGrid: document.getElementById('history-gallery-grid'),
      historyCountBadge: document.getElementById('history-count-badge'),

      // Hardware status banner
      gpuStatusDot: document.getElementById('gpu-status-dot'),
      gpuStatusText: document.getElementById('gpu-status-text'),
      badgeGpuVendor: document.getElementById('badge-gpu-vendor'),
      badgeGpuBuffer: document.getElementById('badge-gpu-buffer'),
      badgeGpuPerf: document.getElementById('badge-gpu-perf')
    };
  }

  bindEvents() {
    // Mode Switcher Tabs
    this.dom.tabModeNeural?.addEventListener('click', () => this.switchMode('neural'));
    this.dom.tabModeBackend?.addEventListener('click', () => this.switchMode('backend'));
    this.dom.tabModeSimulator?.addEventListener('click', () => this.switchMode('simulator'));

    // Download / Load Neural Weights
    this.dom.btnLoadWeights?.addEventListener('click', () => this.handleLoadWeightsClick());
    this.dom.btnClearCache?.addEventListener('click', () => this.handleClearCacheClick());

    // Backend Type Selector
    this.dom.selectBackendType?.addEventListener('change', (e) => {
      const type = e.target.value;
      if (type === 'auto1111') {
        this.dom.groupBackendUrl?.classList.remove('hidden');
        this.dom.groupHfToken?.classList.add('hidden');
      } else if (type === 'hf-api') {
        this.dom.groupBackendUrl?.classList.add('hidden');
        this.dom.groupHfToken?.classList.remove('hidden');
      }
    });

    // Character counter
    this.dom.promptInput?.addEventListener('input', () => {
      const len = this.dom.promptInput.value.length;
      if (this.dom.charCount) this.dom.charCount.textContent = `${len} chars`;
    });

    // Quick Inspiration chips
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
    this.dom.sliderSteps?.addEventListener('input', (e) => {
      const val = e.target.value;
      if (this.dom.valSteps) this.dom.valSteps.textContent = val;
      if (this.dom.stepsBadge) this.dom.stepsBadge.textContent = `${val} Steps`;
    });

    // CFG slider
    this.dom.sliderCfg?.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value).toFixed(1);
      if (this.dom.valCfg) this.dom.valCfg.textContent = val;
    });

    // Resolution
    this.dom.selectResolution?.addEventListener('change', (e) => {
      const r = parseInt(e.target.value, 10);
      if (this.dom.valRes) this.dom.valRes.textContent = `${r}×${r}`;
      if (this.dom.mainCanvas) {
        this.dom.mainCanvas.width = r;
        this.dom.mainCanvas.height = r;
      }
      if (this.dom.attentionCanvas) {
        this.dom.attentionCanvas.width = r;
        this.dom.attentionCanvas.height = r;
      }
    });

    // Randomize seed
    this.dom.btnRandomSeed?.addEventListener('click', () => {
      const newSeed = Math.floor(Math.random() * 900000) + 1000;
      if (this.dom.inputSeed) this.dom.inputSeed.value = newSeed;
    });

    // Generate Button
    this.dom.btnGenerate?.addEventListener('click', () => this.startGeneration());

    // Stop Button
    this.dom.btnCancel?.addEventListener('click', () => {
      if (this.engineMode === 'neural') this.neuralEngine.cancel();
      else if (this.engineMode === 'simulator') this.simEngine.cancel();
    });

    // Download PNG
    this.dom.btnDownloadPng?.addEventListener('click', () => this.downloadCurrentPng());

    // Copy Prompt & Seed
    this.dom.btnCopySettings?.addEventListener('click', () => {
      if (!this.activeGeneration) return;
      const text = `Prompt: "${this.activeGeneration.prompt}"\nSeed: ${this.activeGeneration.seed}\nSteps: ${this.activeGeneration.steps}\nBackend: ${this.activeGeneration.backend}`;
      navigator.clipboard.writeText(text).then(() => {
        this.dom.btnCopySettings.innerHTML = '<span>✅</span> Copied!';
        setTimeout(() => {
          this.dom.btnCopySettings.innerHTML = '<span>📋</span> Copy Seed & Prompt';
        }, 2000);
      });
    });

    // Send to Explainer
    this.dom.btnSendExplainer?.addEventListener('click', () => this.onInspectInExplainer());
  }

  switchMode(mode) {
    this.engineMode = mode;

    this.dom.tabModeNeural?.classList.toggle('active', mode === 'neural');
    this.dom.tabModeBackend?.classList.toggle('active', mode === 'backend');
    this.dom.tabModeSimulator?.classList.toggle('active', mode === 'simulator');

    this.dom.cardNeuralManager?.classList.toggle('hidden', mode !== 'neural');
    this.dom.cardBackendSettings?.classList.toggle('hidden', mode !== 'backend');
    this.dom.cardSimulatorInfo?.classList.toggle('hidden', mode !== 'simulator');

    // Adapt sliders and hints for model type
    if (mode === 'neural') {
      if (this.dom.sliderSteps) {
        this.dom.sliderSteps.max = '4';
        this.dom.sliderSteps.value = '1';
        if (this.dom.valSteps) this.dom.valSteps.textContent = '1';
        if (this.dom.stepsBadge) this.dom.stepsBadge.textContent = '1 Step';
        if (this.dom.stepsHint) this.dom.stepsHint.textContent = '1-4 steps (Distilled SD-Turbo optimal)';
      }
      if (this.dom.sliderCfg) {
        this.dom.sliderCfg.value = '0.0';
        if (this.dom.valCfg) this.dom.valCfg.textContent = '0.0';
        if (this.dom.cfgHint) this.dom.cfgHint.textContent = '0.0 (Optimal for distilled SD-Turbo)';
      }
      if (this.dom.btnGenerateLabel) {
        this.dom.btnGenerateLabel.textContent = this.neuralEngine.isLoaded
          ? 'Generate with Real Neural Net'
          : 'Download Weights & Generate';
      }
    } else if (mode === 'backend') {
      if (this.dom.sliderSteps) {
        this.dom.sliderSteps.max = '50';
        this.dom.sliderSteps.value = '20';
        if (this.dom.valSteps) this.dom.valSteps.textContent = '20';
        if (this.dom.stepsBadge) this.dom.stepsBadge.textContent = '20 Steps';
        if (this.dom.stepsHint) this.dom.stepsHint.textContent = '20-30 steps (Standard SD 1.5 / SDXL)';
      }
      if (this.dom.sliderCfg) {
        this.dom.sliderCfg.value = '7.5';
        if (this.dom.valCfg) this.dom.valCfg.textContent = '7.5';
        if (this.dom.cfgHint) this.dom.cfgHint.textContent = '7.5 (Standard CFG)';
      }
      if (this.dom.btnGenerateLabel) this.dom.btnGenerateLabel.textContent = 'Generate via Server API';
    } else if (mode === 'simulator') {
      if (this.dom.sliderSteps) {
        this.dom.sliderSteps.max = '25';
        this.dom.sliderSteps.value = '15';
        if (this.dom.valSteps) this.dom.valSteps.textContent = '15';
        if (this.dom.stepsBadge) this.dom.stepsBadge.textContent = '15 Steps';
        if (this.dom.stepsHint) this.dom.stepsHint.textContent = '1-25 steps (Interactive mathematical curves)';
      }
      if (this.dom.sliderCfg) {
        this.dom.sliderCfg.value = '7.5';
        if (this.dom.valCfg) this.dom.valCfg.textContent = '7.5';
        if (this.dom.cfgHint) this.dom.cfgHint.textContent = '7.5 (Optimal)';
      }
      if (this.dom.btnGenerateLabel) this.dom.btnGenerateLabel.textContent = 'Run Math Simulator';
    }
  }

  async handleLoadWeightsClick() {
    if (this.neuralEngine.isLoading) return;
    this.dom.btnLoadWeights.disabled = true;
    this.dom.modelProgressContainer?.classList.remove('hidden');

    try {
      await this.neuralEngine.loadNeuralPipeline({
        onProgress: (p) => this.handleDownloadProgress(p)
      });
      this.updateNeuralLoadedUI(true);
    } catch (err) {
      alert(`Model loading error: ${err.message}`);
      this.dom.btnLoadWeights.disabled = false;
    }
  }

  async handleClearCacheClick() {
    if (!confirm('Are you sure you want to delete the cached 2.4 GB ONNX weights from IndexedDB?')) return;
    await this.neuralEngine.clearCache();
    this.updateNeuralLoadedUI(false);
    alert('Model cache cleared.');
  }

  updateNeuralLoadedUI(isLoaded) {
    if (this.dom.badgeModelStatus) {
      this.dom.badgeModelStatus.textContent = isLoaded ? '⚡ WebGPU Ready' : 'Not Loaded';
      this.dom.badgeModelStatus.style.background = isLoaded ? 'rgba(0, 245, 212, 0.2)' : '';
      this.dom.badgeModelStatus.style.color = isLoaded ? 'var(--accent-cyan)' : '';
    }

    if (this.dom.btnLoadWeightsText) {
      this.dom.btnLoadWeightsText.textContent = isLoaded ? 'Model Loaded in GPU Memory' : 'Download & Load Weights (~2.4 GB)';
    }
    if (this.dom.btnLoadWeightsIcon) {
      this.dom.btnLoadWeightsIcon.textContent = isLoaded ? '✅' : '📥';
    }

    if (this.dom.btnLoadWeights) this.dom.btnLoadWeights.disabled = isLoaded;
    this.dom.btnClearCache?.classList.toggle('hidden', !isLoaded);
    this.dom.modelProgressContainer?.classList.add('hidden');

    if (this.dom.btnGenerateLabel && this.engineMode === 'neural') {
      this.dom.btnGenerateLabel.textContent = isLoaded ? 'Generate on WebGPU' : 'Download Weights & Generate';
    }
  }

  handleDownloadProgress(p) {
    if (this.dom.modelProgressContainer) this.dom.modelProgressContainer.classList.remove('hidden');
    if (this.dom.downloadTaskName) {
      this.dom.downloadTaskName.textContent = p.fromCache
        ? `Loaded ${p.name} from cache`
        : `Downloading ${p.name}...`;
    }
    if (this.dom.downloadPct) this.dom.downloadPct.textContent = `${p.percent}%`;
    if (this.dom.downloadProgressFill) this.dom.downloadProgressFill.style.width = `${p.percent}%`;

    const loadedMB = (p.loadedBytes / (1024 * 1024)).toFixed(0);
    const totalMB = (p.totalBytes / (1024 * 1024)).toFixed(0);
    if (this.dom.downloadBytesInfo) {
      this.dom.downloadBytesInfo.textContent = `${loadedMB} MB / ${totalMB} MB`;
    }
    if (this.dom.downloadSpeedInfo) {
      this.dom.downloadSpeedInfo.textContent = p.fromCache
        ? 'Cached locally in IndexedDB'
        : `${p.speedMBs} MB/s • ETA: ${p.etaSec}s`;
    }
  }

  handleNeuralStatus({ status, message, cacheStatus }) {
    if (this.dom.gpuStatusText) this.dom.gpuStatusText.textContent = message;

    if (cacheStatus) {
      const allCached = cacheStatus.allCached;
      if (this.dom.badgeModelStatus) {
        this.dom.badgeModelStatus.textContent = allCached ? 'Cached locally (IndexedDB)' : 'Needs Download (~2.4 GB)';
      }
      this.dom.btnClearCache?.classList.toggle('hidden', !allCached);
    }
  }

  handleSimStatus({ status, message, backend, deviceInfo }) {
    if (this.dom.gpuStatusDot) {
      this.dom.gpuStatusDot.className = (backend === 'webgpu')
        ? 'gpu-status-dot dot-active'
        : 'gpu-status-dot dot-warning';
    }
    if (deviceInfo) {
      if (this.dom.badgeGpuVendor) this.dom.badgeGpuVendor.textContent = deviceInfo.vendor || 'GPU Active';
      if (this.dom.badgeGpuBuffer) this.dom.badgeGpuBuffer.textContent = `Max Buffer: ${deviceInfo.maxBufferSizeMB} MB`;
      if (this.dom.badgeGpuPerf) this.dom.badgeGpuPerf.textContent = 'WebGPU Available';
    }
  }

  async startGeneration() {
    const prompt = this.dom.promptInput ? this.dom.promptInput.value.trim() : '';
    const negativePrompt = this.dom.negativePromptInput ? this.dom.negativePromptInput.value.trim() : '';
    const steps = parseInt(this.dom.sliderSteps.value, 10) || 1;
    const guidanceScale = parseFloat(this.dom.sliderCfg.value) || 0.0;
    const seed = this.dom.inputSeed.value !== '' ? parseInt(this.dom.inputSeed.value, 10) : Math.floor(Math.random() * 1000000);
    const size = parseInt(this.dom.selectResolution.value, 10) || 512;

    if (!prompt) {
      alert('Please enter a prompt to generate!');
      return;
    }

    this.dom.btnGenerate.disabled = true;
    this.dom.btnGenerateLabel.textContent = 'Generating...';
    this.dom.btnCancel.classList.remove('hidden');
    if (this.dom.progressBar) this.dom.progressBar.style.width = '0%';

    const mainCtx = this.dom.mainCanvas.getContext('2d');
    const latentCtx = this.dom.latentCanvas.getContext('2d');

    try {
      if (this.engineMode === 'neural') {
        // Mode 1: Real In-Browser WebGPU Neural Network
        if (!this.neuralEngine.isLoaded) {
          if (confirm('The genuine SD-Turbo model (~2.4 GB) will now be downloaded and cached into your browser\'s IndexedDB. Proceed?')) {
            await this.handleLoadWeightsClick();
          } else {
            this.dom.btnGenerate.disabled = false;
            this.dom.btnGenerateLabel.textContent = 'Generate on WebGPU';
            this.dom.btnCancel.classList.add('hidden');
            return;
          }
        }

        if (this.dom.liveStepStatus) this.dom.liveStepStatus.textContent = 'Running CLIP & U-Net on WebGPU...';

        const result = await this.neuralEngine.generate({
          prompt,
          steps,
          seed,
          onStep: (stepInfo) => {
            const pct = Math.round((stepInfo.currentStep / stepInfo.totalSteps) * 100);
            if (this.dom.progressBar) this.dom.progressBar.style.width = `${pct}%`;
            if (this.dom.liveStepStatus) {
              this.dom.liveStepStatus.textContent = `Neural Step ${stepInfo.currentStep}/${stepInfo.totalSteps} (t=${stepInfo.timestep})`;
            }
            if (this.dom.liveLatencyStatus) {
              this.dom.liveLatencyStatus.textContent = `${Math.round(stepInfo.stepLatencyMs)}ms`;
            }
          },
          onComplete: (result) => {
            this.activeGeneration = result;
            mainCtx.drawImage(result.canvas, 0, 0, size, size);
            latentCtx.drawImage(result.canvas, 0, 0, 64, 64);

            if (this.dom.btnDownloadPng) this.dom.btnDownloadPng.disabled = false;
            if (this.dom.btnCopySettings) this.dom.btnCopySettings.disabled = false;
            this.addToHistory(result);

            if (this.dom.liveStepStatus) {
              const timeSec = (result.totalTimeMs / 1000).toFixed(1);
              this.dom.liveStepStatus.textContent = `Completed (Real WebGPU Neural Net) in ${timeSec}s`;
            }
          }
        });
      } else if (this.engineMode === 'backend') {
        // Mode 2: Direct Server API
        const type = this.dom.selectBackendType.value;
        const url = this.dom.inputBackendUrl.value.trim();
        const token = this.dom.inputHfToken.value.trim();
        this.serverEngine.setBackend(type, url, token);

        if (this.dom.liveStepStatus) this.dom.liveStepStatus.textContent = `Calling ${type.toUpperCase()} API...`;

        const result = await this.serverEngine.generate({
          prompt,
          negativePrompt,
          steps,
          guidanceScale,
          seed,
          width: size,
          height: size
        });

        this.activeGeneration = result;
        mainCtx.drawImage(result.canvas, 0, 0, size, size);
        latentCtx.drawImage(result.canvas, 0, 0, 64, 64);
        if (this.dom.btnDownloadPng) this.dom.btnDownloadPng.disabled = false;
        if (this.dom.btnCopySettings) this.dom.btnCopySettings.disabled = false;
        this.addToHistory(result);

        if (this.dom.liveStepStatus) {
          const timeSec = (result.totalTimeMs / 1000).toFixed(1);
          this.dom.liveStepStatus.textContent = `Completed (${result.backend}) in ${timeSec}s`;
        }
      } else {
        // Mode 3: Educational Simulator
        await this.simEngine.generate({
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
              this.dom.liveStepStatus.textContent = `Step ${stepInfo.currentStep}/${stepInfo.totalSteps} (t=${stepInfo.timestep})`;
            }
            if (this.dom.liveLatencyStatus) {
              this.dom.liveLatencyStatus.textContent = `${Math.round(stepInfo.stepLatencyMs)}ms`;
            }
            if (stepInfo.frameCanvas) mainCtx.drawImage(stepInfo.frameCanvas, 0, 0, size, size);
            if (this.simEngine.latentCanvas) latentCtx.drawImage(this.simEngine.latentCanvas, 0, 0, 64, 64);

            if (!this.tokenData && stepInfo.tokens) {
              this.tokenData = { tokens: stepInfo.tokens, hotspots: stepInfo.hotspots };
              this.renderTokenChips(stepInfo.tokens, stepInfo.hotspots);
            }
          },
          onComplete: (result) => {
            this.activeGeneration = result;
            if (this.dom.btnDownloadPng) this.dom.btnDownloadPng.disabled = false;
            if (this.dom.btnCopySettings) this.dom.btnCopySettings.disabled = false;
            this.addToHistory(result);

            if (this.dom.liveStepStatus) {
              const timeSec = (result.totalTimeMs / 1000).toFixed(1);
              this.dom.liveStepStatus.textContent = `Completed in ${timeSec}s`;
            }
          }
        });
      }
    } catch (err) {
      console.error('Generation error:', err);
      alert(`Generation failed: ${err.message}`);
      if (this.dom.liveStepStatus) this.dom.liveStepStatus.textContent = 'Error during generation';
    } finally {
      this.dom.btnGenerate.disabled = false;
      this.dom.btnCancel.classList.add('hidden');
      if (this.dom.btnGenerateLabel) {
        if (this.engineMode === 'neural') {
          this.dom.btnGenerateLabel.textContent = this.neuralEngine.isLoaded
            ? 'Generate on WebGPU'
            : 'Download Weights & Generate';
        } else if (this.engineMode === 'backend') {
          this.dom.btnGenerateLabel.textContent = 'Generate via Server API';
        } else {
          this.dom.btnGenerateLabel.textContent = 'Run Math Simulator';
        }
      }
    }
  }

  renderTokenChips(tokens, hotspots) {
    if (!this.dom.tokensList || !tokens) return;

    this.dom.tokensList.innerHTML = tokens.map(t => {
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
      backend: result.backend,
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
      <div class="history-item" data-id="${item.id}" title="${escapeHtml(item.prompt)} (Seed: ${item.seed} • ${item.backend})">
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
          if (this.dom.promptInput) this.dom.promptInput.value = match.prompt;
          if (this.dom.inputSeed) this.dom.inputSeed.value = match.seed;
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
    link.download = `stable-diffusion-seed-${this.activeGeneration?.seed || 'gen'}.png`;
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

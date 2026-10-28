/**
 * Diffusion Explainer - Main Application Controller
 * Orchestrates text tokenization, 4-channel latent space rendering,
 * cross-attention heatmaps, timestep scrubbing animation, noise schedules,
 * and Classifier-Free Guidance exploration.
 */

import { PRESETS } from './data/presets.js';
import { PromptTokenizer } from './components/tokenizer.js';
import { LatentRenderer } from './components/latentRenderer.js';
import { CrossAttentionManager } from './components/crossAttention.js';
import { SchedulerChart } from './components/schedulerChart.js';
import { CFGExplorer } from './components/cfgExplorer.js';
import { UNetDiagram } from './components/unetDiagram.js';
import { WebGPUStudio } from './components/webgpuStudio.js';
import { COLORMAPS } from './utils/colormaps.js';
import { getScheduleParams } from './utils/math.js';

class DiffusionExplainerApp {
  constructor() {
    this.presets = PRESETS;
    this.currentPresetIndex = 0;
    this.currentStep = 50; // 50 (noise) down to 0 (clean image)
    this.totalSteps = 50;
    this.isPlaying = false;
    this.playbackInterval = null;
    this.playbackSpeedMs = 120; // 120ms per step
    this.colormap = COLORMAPS.TURBO;

    // Off-screen canvas to extract 64x64 clean image pixels for latent synthesis
    this.sample64Canvas = document.createElement('canvas');
    this.sample64Canvas.width = 64;
    this.sample64Canvas.height = 64;
    this.sample64Ctx = this.sample64Canvas.getContext('2d', { willReadFrequently: true });
    this.clean64ImageData = null;

    // Image preloader cache
    this.imageCache = new Map();
  }

  async init() {
    this.cacheDOMElements();
    await this.initSubcomponents();
    this.bindEvents();

    // Preload current preset trajectory images
    await this.loadPreset(0);

    // Initial render at step 50 (pure noise) or step 25 (midpoint)
    this.setStep(50);
  }

  cacheDOMElements() {
    this.dom = {
      // Main Application Tabs
      tabBtnExplainer: document.getElementById('tab-btn-explainer'),
      tabBtnWebgpu: document.getElementById('tab-btn-webgpu'),
      viewExplainer: document.getElementById('view-explainer'),
      viewWebgpu: document.getElementById('view-webgpu'),

      // Explainer Elements
      presetCardsContainer: document.getElementById('preset-cards-container'),
      activePromptDisplay: document.getElementById('active-prompt-display'),
      mainDecodedImage: document.getElementById('main-decoded-image'),
      stepSlider: document.getElementById('step-slider'),
      btnPlayPause: document.getElementById('btn-play-pause'),
      btnStepPrev: document.getElementById('btn-step-prev'),
      btnStepNext: document.getElementById('btn-step-next'),
      btnReset: document.getElementById('btn-reset'),
      stepLabel: document.getElementById('step-label'),
      timestepBadge: document.getElementById('timestep-badge'),
      noisePctBadge: document.getElementById('noise-pct-badge'),
      signalPctBadge: document.getElementById('signal-pct-badge'),
      colormapSelect: document.getElementById('colormap-select'),
      latentModeSelect: document.getElementById('latent-mode-select'),
      toggleAttention: document.getElementById('toggle-attention'),
      sliderAttAlpha: document.getElementById('slider-att-alpha'),
      alphaValDisplay: document.getElementById('alpha-val-display'),
      speedSelect: document.getElementById('speed-select')
    };
  }

  async initSubcomponents() {
    // 1. Text Tokenizer
    this.tokenizer = new PromptTokenizer({
      containerId: 'tokenizer-container',
      onTokenSelect: (tokenData) => {
        if (this.crossAttention) {
          this.crossAttention.setTokenAttention(tokenData.token, tokenData.hotspot);
        }
      }
    });

    // 2. Latent Visualizer (64x64x4)
    this.latentRenderer = new LatentRenderer({
      canvasId: 'latent-canvas',
      statsId: 'latent-stats',
      colormap: this.colormap
    });

    // 3. Cross-Attention Heatmap Manager
    this.crossAttention = new CrossAttentionManager({
      canvasId: 'attention-canvas-overlay',
      colormap: this.colormap
    });

    // 4. Noise Scheduler SVG Chart
    this.schedulerChart = new SchedulerChart({
      containerId: 'scheduler-chart-container',
      onSeekStep: (step) => {
        this.setStep(step);
      }
    });

    // 5. Classifier-Free Guidance Explorer
    this.cfgExplorer = new CFGExplorer({
      containerId: 'cfg-explorer-container',
      onScaleChange: (scale) => {
        // Can optionally influence contrast or saturation
      }
    });

    // 6. U-Net Architecture Diagram
    this.unetDiagram = new UNetDiagram({
      containerId: 'unet-diagram-container'
    });

    // 7. Live WebGPU Studio Tab
    this.webgpuStudio = new WebGPUStudio({
      containerId: 'webgpu-studio-container',
      onInspectInExplainer: () => {
        this.switchTab('explainer');
      }
    });
    await this.webgpuStudio.init();
  }

  switchTab(tabName) {
    if (tabName === 'explainer') {
      if (this.dom.tabBtnExplainer) this.dom.tabBtnExplainer.classList.add('active');
      if (this.dom.tabBtnWebgpu) this.dom.tabBtnWebgpu.classList.remove('active');
      if (this.dom.viewExplainer) this.dom.viewExplainer.classList.remove('hidden');
      if (this.dom.viewWebgpu) this.dom.viewWebgpu.classList.add('hidden');
    } else if (tabName === 'webgpu') {
      if (this.dom.tabBtnWebgpu) this.dom.tabBtnWebgpu.classList.add('active');
      if (this.dom.tabBtnExplainer) this.dom.tabBtnExplainer.classList.remove('active');
      if (this.dom.viewWebgpu) this.dom.viewWebgpu.classList.remove('hidden');
      if (this.dom.viewExplainer) this.dom.viewExplainer.classList.add('hidden');
    }
  }

  bindEvents() {
    // Navigation Tabs Switching (Explainer vs WebGPU Studio)
    if (this.dom.tabBtnExplainer) {
      this.dom.tabBtnExplainer.addEventListener('click', () => {
        this.switchTab('explainer');
      });
    }

    if (this.dom.tabBtnWebgpu) {
      this.dom.tabBtnWebgpu.addEventListener('click', () => {
        this.switchTab('webgpu');
      });
    }

    // Timestep Scrubber Slider
    if (this.dom.stepSlider) {
      this.dom.stepSlider.addEventListener('input', (e) => {
        this.pause();
        this.setStep(parseInt(e.target.value, 10));
      });
    }

    // Play/Pause Button
    if (this.dom.btnPlayPause) {
      this.dom.btnPlayPause.addEventListener('click', () => {
        if (this.isPlaying) {
          this.pause();
        } else {
          this.play();
        }
      });
    }

    // Step Forward / Back Buttons
    if (this.dom.btnStepPrev) {
      this.dom.btnStepPrev.addEventListener('click', () => {
        this.pause();
        this.setStep(Math.min(50, this.currentStep + 5));
      });
    }

    if (this.dom.btnStepNext) {
      this.dom.btnStepNext.addEventListener('click', () => {
        this.pause();
        this.setStep(Math.max(0, this.currentStep - 5));
      });
    }

    if (this.dom.btnReset) {
      this.dom.btnReset.addEventListener('click', () => {
        this.pause();
        this.setStep(50);
      });
    }

    // Playback Speed
    if (this.dom.speedSelect) {
      this.dom.speedSelect.addEventListener('change', (e) => {
        this.playbackSpeedMs = parseInt(e.target.value, 10);
        if (this.isPlaying) {
          this.pause();
          this.play();
        }
      });
    }

    // Colormap Selector
    if (this.dom.colormapSelect) {
      this.dom.colormapSelect.addEventListener('change', (e) => {
        this.colormap = e.target.value;
        this.latentRenderer.setColormap(this.colormap);
        this.crossAttention.setColormap(this.colormap);
      });
    }

    // Latent View Mode (PCA vs Channels vs Grid)
    if (this.dom.latentModeSelect) {
      this.dom.latentModeSelect.addEventListener('change', (e) => {
        this.latentRenderer.setViewMode(e.target.value);
      });
    }

    // Cross-Attention Toggle & Alpha
    if (this.dom.toggleAttention) {
      this.dom.toggleAttention.addEventListener('change', (e) => {
        this.crossAttention.setEnabled(e.target.checked);
      });
    }

    if (this.dom.sliderAttAlpha) {
      this.dom.sliderAttAlpha.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        this.crossAttention.setAlpha(val);
        if (this.dom.alphaValDisplay) {
          this.dom.alphaValDisplay.textContent = `${Math.round(val * 100)}%`;
        }
      });
    }

    // Quick Milestone Jump Buttons
    document.querySelectorAll('.milestone-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.pause();
        const step = parseInt(btn.dataset.step, 10);
        this.setStep(step);
      });
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        this.isPlaying ? this.pause() : this.play();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        this.pause();
        this.setStep(Math.min(50, this.currentStep + 1));
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        this.pause();
        this.setStep(Math.max(0, this.currentStep - 1));
      } else if (e.code === 'Home') {
        e.preventDefault();
        this.pause();
        this.setStep(50);
      } else if (e.code === 'End') {
        e.preventDefault();
        this.pause();
        this.setStep(0);
      }
    });

    // Render Preset Prompt Cards
    this.renderPresetSelector();
  }

  renderPresetSelector() {
    if (!this.dom.presetCardsContainer) return;

    this.dom.presetCardsContainer.innerHTML = this.presets.map((preset, idx) => {
      const isSelected = idx === this.currentPresetIndex;
      return `
        <button 
          class="preset-card ${isSelected ? 'active' : ''}" 
          data-idx="${idx}"
          type="button"
        >
          <div class="preset-thumb-wrap">
            <img class="preset-thumb" src="${preset.final_image}" alt="${preset.name}" loading="lazy">
          </div>
          <div class="preset-info">
            <div class="preset-title">${preset.name}</div>
            <div class="preset-prompt-snippet">"${preset.prompt}"</div>
          </div>
        </button>
      `;
    }).join('');

    this.dom.presetCardsContainer.querySelectorAll('.preset-card').forEach(card => {
      card.addEventListener('click', async () => {
        const idx = parseInt(card.dataset.idx, 10);
        await this.loadPreset(idx);
      });
    });
  }

  async loadPreset(index) {
    this.currentPresetIndex = index;
    const preset = this.presets[index];
    if (!preset) return;

    // Update preset cards UI
    if (this.dom.presetCardsContainer) {
      this.dom.presetCardsContainer.querySelectorAll('.preset-card').forEach((c, i) => {
        c.classList.toggle('active', i === index);
      });
    }

    if (this.dom.activePromptDisplay) {
      this.dom.activePromptDisplay.textContent = `"${preset.prompt}"`;
    }

    // Preload images into browser memory
    preset.steps.forEach(st => {
      if (!this.imageCache.has(st.image_url)) {
        const img = new Image();
        img.src = st.image_url;
        this.imageCache.set(st.image_url, img);
      }
    });

    // Extract 64x64 clean target image representation for latent synthesis
    const finalImg = new Image();
    finalImg.src = preset.final_image;
    await new Promise((resolve) => {
      if (finalImg.complete) {
        this.sample64Ctx.drawImage(finalImg, 0, 0, 64, 64);
        this.clean64ImageData = this.sample64Ctx.getImageData(0, 0, 64, 64).data;
        resolve();
      } else {
        finalImg.onload = () => {
          this.sample64Ctx.drawImage(finalImg, 0, 0, 64, 64);
          this.clean64ImageData = this.sample64Ctx.getImageData(0, 0, 64, 64).data;
          resolve();
        };
      }
    });

    // Update Tokenizer with new prompt tokens & attention hotspots
    this.tokenizer.setPromptData(preset.tokens, preset.attention_hotspots);

    // Refresh current step
    this.setStep(this.currentStep);
  }

  setStep(step) {
    this.currentStep = Math.max(0, Math.min(this.totalSteps, step));

    // 1. Update Slider UI
    if (this.dom.stepSlider) {
      this.dom.stepSlider.value = this.currentStep;
    }

    const params = getScheduleParams(this.currentStep, this.totalSteps);

    // 2. Update Header & Badges
    if (this.dom.stepLabel) {
      this.dom.stepLabel.textContent = `Step ${this.totalSteps - this.currentStep} of ${this.totalSteps}`;
    }
    if (this.dom.timestepBadge) {
      this.dom.timestepBadge.textContent = `Timestep t=${params.timestep}`;
    }
    if (this.dom.noisePctBadge) {
      this.dom.noisePctBadge.textContent = `Noise: ${(params.sigma * 100).toFixed(1)}%`;
    }
    if (this.dom.signalPctBadge) {
      this.dom.signalPctBadge.textContent = `Signal: ${(params.alphaBar * 100).toFixed(1)}%`;
    }

    // 3. Update Decoded Image Frame
    const preset = this.presets[this.currentPresetIndex];
    if (preset && this.dom.mainDecodedImage) {
      // Find closest milestone frame
      const frameObj = this.findClosestMilestoneFrame(preset.steps, this.currentStep);
      if (frameObj) {
        this.dom.mainDecodedImage.src = frameObj.image_url;
      }
    }

    // 4. Update Latent Tensor Canvas (64x64x4)
    this.latentRenderer.updateLatents(this.clean64ImageData, this.currentStep, this.totalSteps, this.currentPresetIndex * 997);

    // 5. Update Scheduler SVG Tracking Cursor
    this.schedulerChart.setStep(this.currentStep);
  }

  findClosestMilestoneFrame(steps, targetStep) {
    let closest = steps[0];
    let minDiff = Infinity;
    for (const s of steps) {
      const diff = Math.abs(s.step - targetStep);
      if (diff < minDiff) {
        minDiff = diff;
        closest = s;
      }
    }
    return closest;
  }

  play() {
    this.isPlaying = true;
    if (this.dom.btnPlayPause) {
      this.dom.btnPlayPause.innerHTML = '⏸️ Pause';
    }

    if (this.currentStep === 0) {
      this.setStep(50);
    }

    clearInterval(this.playbackInterval);
    this.playbackInterval = setInterval(() => {
      if (this.currentStep > 0) {
        this.setStep(this.currentStep - 1);
      } else {
        this.pause();
      }
    }, this.playbackSpeedMs);
  }

  pause() {
    this.isPlaying = false;
    clearInterval(this.playbackInterval);
    if (this.dom.btnPlayPause) {
      this.dom.btnPlayPause.innerHTML = '▶️ Play Denoising';
    }
  }
}

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new DiffusionExplainerApp();
  app.init();
  window.__diffusionApp = app;
});

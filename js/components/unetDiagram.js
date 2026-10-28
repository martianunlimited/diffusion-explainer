/**
 * Interactive Stable Diffusion U-Net & Cross-Attention Architecture Diagram
 * Visualizes the 860M parameter U-Net: Encoder, Bottleneck, Decoder,
 * Skip Connections, Timestep Conditioning, and Text Cross-Attention injection.
 */

export const UNET_BLOCKS = {
  input_latent: {
    title: 'Input Noisy Latent zₜ',
    tensor: 'Tensor: [1, 4, 64, 64]',
    desc: 'The current latent representation at timestep t. 4 channels compress 512×512 RGB space by 48×.'
  },
  time_embed: {
    title: 'Timestep Embedding (t)',
    tensor: 'Tensor: [1, 1280]',
    desc: 'Sinusoidal positional encoding followed by an MLP. Informs each ResNet block how much noise remains to remove.'
  },
  text_embed: {
    title: 'CLIP Text Condition (K, V)',
    tensor: 'Tensor: [1, 77, 768]',
    desc: 'The 77-token sequence from CLIP ViT-L/14 text encoder. Serves as Keys and Values in all Cross-Attention layers.'
  },
  down1: {
    title: 'Down-Block 1 (ResNet + Cross-Attn)',
    tensor: 'Resolution: 64×64 • Channels: 320',
    desc: 'Extracts fine spatial primitives and early composition. Spatial queries attend to prompt tokens.'
  },
  down2: {
    title: 'Down-Block 2 (Downsample 2×)',
    tensor: 'Resolution: 32×32 • Channels: 640',
    desc: 'Downsamples spatial dimensions while doubling channel depth. Captures mid-level textures and structures.'
  },
  down3: {
    title: 'Down-Block 3 (Downsample 2×)',
    tensor: 'Resolution: 16×16 • Channels: 1280',
    desc: 'Deep feature extraction. Integrates higher-level object parts with prompt keywords.'
  },
  mid: {
    title: 'Middle Bottleneck Block',
    tensor: 'Resolution: 8×8 • Channels: 1280',
    desc: 'Highest semantic abstraction. Contains ResNet + Self-Attention (global spatial coherence) + Cross-Attention (core subject focus).'
  },
  up3: {
    title: 'Up-Block 3 (Upsample 2× + Skip)',
    tensor: 'Resolution: 16×16 • Channels: 1280',
    desc: 'Concatenates skip-connection from Down-Block 3 to recover spatial fidelity. Cross-attention refines object outlines.'
  },
  up2: {
    title: 'Up-Block 2 (Upsample 2× + Skip)',
    tensor: 'Resolution: 32×32 • Channels: 640',
    desc: 'Upsamples to 32×32. Blends deep semantic guidance with fine spatial features from Down-Block 2.'
  },
  up1: {
    title: 'Up-Block 1 (Upsample 2× + Skip)',
    tensor: 'Resolution: 64×64 • Channels: 320',
    desc: 'Restores the original 64×64 latent resolution. Adds skip-connection features from Down-Block 1 for crisp edges.'
  },
  output_noise: {
    title: 'Predicted Latent Noise ε_θ',
    tensor: 'Tensor: [1, 4, 64, 64]',
    desc: 'The U-Net output! Represents the noise vector to subtract from zₜ using the scheduler equation to yield z_{t-1}.'
  }
};

export class UNetDiagram {
  constructor({ containerId }) {
    this.container = document.getElementById(containerId);
    this.selectedBlockKey = 'mid';
    this.init();
  }

  init() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="unet-diagram-card">
        <div class="unet-header">
          <div class="unet-title">
            <span class="badge-icon">🧬</span>
            <span>U-Net Denoiser Backbone (860M Parameters)</span>
          </div>
          <div class="unet-sub">Click any architectural block to inspect tensor shapes and operations</div>
        </div>

        <div class="unet-interactive-stage">
          <svg viewBox="0 0 760 300" class="unet-svg" id="unet-svg">
            <defs>
              <linearGradient id="grad-down" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#4361ee" />
                <stop offset="100%" stop-color="#3a0ca3" />
              </linearGradient>
              <linearGradient id="grad-mid" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#f72585" />
                <stop offset="100%" stop-color="#7209b7" />
              </linearGradient>
              <linearGradient id="grad-up" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#00f5d4" />
                <stop offset="100%" stop-color="#058c7e" />
              </linearGradient>
              <linearGradient id="grad-cond" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#f59e0b" />
                <stop offset="100%" stop-color="#b45309" />
              </linearGradient>
            </defs>

            <!-- Conditioning Injections from Top -->
            <g class="unet-node" data-key="time_embed" transform="translate(180, 20)">
              <rect width="180" height="34" rx="6" fill="url(#grad-cond)" stroke="#f59e0b" stroke-width="1.5" />
              <text x="90" y="21" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">⏱️ Timestep Emb [1, 1280]</text>
            </g>

            <g class="unet-node" data-key="text_embed" transform="translate(400, 20)">
              <rect width="180" height="34" rx="6" fill="url(#grad-cond)" stroke="#f59e0b" stroke-width="1.5" />
              <text x="90" y="21" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">💬 CLIP Text [1, 77, 768]</text>
            </g>

            <!-- Condition injection lines -->
            <line x1="270" y1="54" x2="270" y2="90" stroke="#f59e0b" stroke-width="2" stroke-dasharray="3,3" />
            <line x1="490" y1="54" x2="490" y2="90" stroke="#f59e0b" stroke-width="2" stroke-dasharray="3,3" />

            <!-- Input Node -->
            <g class="unet-node" data-key="input_latent" transform="translate(20, 110)">
              <rect width="110" height="50" rx="8" fill="#141e33" stroke="#00f5d4" stroke-width="2" />
              <text x="55" y="24" text-anchor="middle" fill="#00f5d4" font-size="11" font-weight="700">Input Latent</text>
              <text x="55" y="40" text-anchor="middle" fill="#94a3b8" font-size="9" font-family="monospace">zₜ [1,4,64,64]</text>
            </g>
            <line x1="130" y1="135" x2="160" y2="135" stroke="#94a3b8" stroke-width="2" />

            <!-- Downsampling Path (Left branch of U) -->
            <g class="unet-node" data-key="down1" transform="translate(160, 100)">
              <rect width="110" height="60" rx="6" fill="url(#grad-down)" stroke="#4361ee" stroke-width="1.5" />
              <text x="55" y="26" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="700">Down 1 (64×64)</text>
              <text x="55" y="44" text-anchor="middle" fill="#cbd5e1" font-size="8.5" font-family="monospace">320 channels</text>
            </g>

            <line x1="215" y1="160" x2="215" y2="185" stroke="#4361ee" stroke-width="2" />

            <g class="unet-node" data-key="down2" transform="translate(160, 185)">
              <rect width="110" height="45" rx="6" fill="url(#grad-down)" stroke="#4361ee" stroke-width="1.5" />
              <text x="55" y="22" text-anchor="middle" fill="#ffffff" font-size="10" font-weight="700">Down 2 (32×32)</text>
              <text x="55" y="36" text-anchor="middle" fill="#cbd5e1" font-size="8.5" font-family="monospace">640 channels</text>
            </g>

            <line x1="215" y1="230" x2="215" y2="250" stroke="#4361ee" stroke-width="2" />

            <g class="unet-node" data-key="down3" transform="translate(160, 250)">
              <rect width="110" height="40" rx="6" fill="url(#grad-down)" stroke="#4361ee" stroke-width="1.5" />
              <text x="55" y="24" text-anchor="middle" fill="#ffffff" font-size="9.5" font-weight="700">Down 3 (16×16)</text>
            </g>

            <path d="M 270 270 L 320 270" stroke="#7209b7" stroke-width="2.5" />

            <!-- Middle Bottleneck Block -->
            <g class="unet-node active" data-key="mid" transform="translate(320, 240)">
              <rect width="120" height="55" rx="8" fill="url(#grad-mid)" stroke="#f72585" stroke-width="2.5" />
              <text x="60" y="24" text-anchor="middle" fill="#ffffff" font-size="11" font-weight="700">Bottleneck (8×8)</text>
              <text x="60" y="42" text-anchor="middle" fill="#fbcfe8" font-size="9" font-family="monospace">Self + Cross-Attn</text>
            </g>

            <path d="M 440 270 L 490 270" stroke="#7209b7" stroke-width="2.5" />

            <!-- Upsampling Path (Right branch of U) -->
            <g class="unet-node" data-key="up3" transform="translate(490, 250)">
              <rect width="110" height="40" rx="6" fill="url(#grad-up)" stroke="#00f5d4" stroke-width="1.5" />
              <text x="55" y="24" text-anchor="middle" fill="#000000" font-size="9.5" font-weight="700">Up 3 (16×16)</text>
            </g>

            <line x1="545" y1="250" x2="545" y2="230" stroke="#00f5d4" stroke-width="2" />

            <g class="unet-node" data-key="up2" transform="translate(490, 185)">
              <rect width="110" height="45" rx="6" fill="url(#grad-up)" stroke="#00f5d4" stroke-width="1.5" />
              <text x="55" y="22" text-anchor="middle" fill="#000000" font-size="10" font-weight="700">Up 2 (32×32)</text>
              <text x="55" y="36" text-anchor="middle" fill="#042f2e" font-size="8.5" font-family="monospace">640 channels</text>
            </g>

            <line x1="545" y1="185" x2="545" y2="160" stroke="#00f5d4" stroke-width="2" />

            <g class="unet-node" data-key="up1" transform="translate(490, 100)">
              <rect width="110" height="60" rx="6" fill="url(#grad-up)" stroke="#00f5d4" stroke-width="1.5" />
              <text x="55" y="26" text-anchor="middle" fill="#000000" font-size="10" font-weight="700">Up 1 (64×64)</text>
              <text x="55" y="44" text-anchor="middle" fill="#042f2e" font-size="8.5" font-family="monospace">320 channels</text>
            </g>

            <!-- Skip Connections (Horizontal Dashed Lines) -->
            <path d="M 270 130 C 370 130, 390 130, 490 130" stroke="#00f5d4" stroke-width="1.5" stroke-dasharray="4,4" fill="none" opacity="0.6" />
            <path d="M 270 205 C 370 205, 390 205, 490 205" stroke="#00f5d4" stroke-width="1.5" stroke-dasharray="4,4" fill="none" opacity="0.6" />
            <path d="M 270 270 C 370 270, 390 270, 490 270" stroke="#00f5d4" stroke-width="1.5" stroke-dasharray="4,4" fill="none" opacity="0.6" />
            <text x="380" y="125" text-anchor="middle" fill="#00f5d4" font-size="8.5" font-family="monospace">Skip 1</text>
            <text x="380" y="200" text-anchor="middle" fill="#00f5d4" font-size="8.5" font-family="monospace">Skip 2</text>

            <!-- Output Node -->
            <line x1="600" y1="130" x2="630" y2="130" stroke="#94a3b8" stroke-width="2" />
            <g class="unet-node" data-key="output_noise" transform="translate(630, 105)">
              <rect width="115" height="50" rx="8" fill="#141e33" stroke="#f72585" stroke-width="2" />
              <text x="57" y="24" text-anchor="middle" fill="#f72585" font-size="11" font-weight="700">Predicted Noise</text>
              <text x="57" y="40" text-anchor="middle" fill="#cbd5e1" font-size="9" font-family="monospace">ε_θ [1,4,64,64]</text>
            </g>
          </svg>
        </div>

        <!-- Node Details Popover / Card -->
        <div class="unet-detail-card" id="unet-detail-card">
          ${this.renderBlockDetails(this.selectedBlockKey)}
        </div>
      </div>
    `;

    // Click handler for nodes
    this.container.querySelectorAll('.unet-node').forEach(node => {
      node.style.cursor = 'pointer';
      node.addEventListener('click', () => {
        this.container.querySelectorAll('.unet-node').forEach(n => n.classList.remove('active'));
        node.classList.add('active');
        this.selectedBlockKey = node.dataset.key;
        const detailCard = document.getElementById('unet-detail-card');
        if (detailCard) {
          detailCard.innerHTML = this.renderBlockDetails(this.selectedBlockKey);
        }
      });
    });
  }

  renderBlockDetails(key) {
    const info = UNET_BLOCKS[key] || UNET_BLOCKS.mid;
    return `
      <div class="block-detail-content">
        <div class="block-detail-title-row">
          <span class="block-detail-name">${info.title}</span>
          <span class="block-detail-tensor">${info.tensor}</span>
        </div>
        <div class="block-detail-desc">${info.desc}</div>
      </div>
    `;
  }
}

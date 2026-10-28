/**
 * Interactive CLIP Prompt Tokenizer Component
 * Breaks prompts into tokens, visualizes token embeddings (768-D),
 * and dispatches token selection events for cross-attention heatmaps.
 */

export class PromptTokenizer {
  constructor({ containerId, onTokenSelect }) {
    this.container = document.getElementById(containerId);
    this.onTokenSelect = onTokenSelect || (() => {});
    this.selectedToken = null;
    this.tokens = [];
    this.hotspots = {};
  }

  setPromptData(tokens, hotspots, initialSelected = null) {
    this.tokens = tokens || [];
    this.hotspots = hotspots || {};
    this.selectedToken = initialSelected || (this.findDefaultHotspotToken() || this.tokens[1] || null);
    this.render();
  }

  findDefaultHotspotToken() {
    for (const key of Object.keys(this.hotspots)) {
      const match = this.tokens.find(t => key.toLowerCase().includes(t.toLowerCase()) || t.toLowerCase().includes(key.toLowerCase()));
      if (match) return match;
    }
    return null;
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="tokenizer-wrapper">
        <div class="tokenizer-header">
          <div class="tokenizer-title">
            <span class="badge-icon">🔤</span>
            <span>CLIP Text Tokenizer (77-Token Context)</span>
          </div>
          <div class="tokenizer-hint">Click any token to inspect its Cross-Attention spatial heatmap</div>
        </div>
        <div class="tokens-chip-list" id="tokens-chip-list">
          ${this.tokens.map((token, idx) => {
            const hasHotspot = this.hasHotspotForToken(token);
            const isSelected = this.selectedToken === token;
            const isSpecial = token.startsWith('<') && token.endsWith('>');

            return `
              <button 
                class="token-chip ${isSelected ? 'active' : ''} ${hasHotspot ? 'has-attention' : ''} ${isSpecial ? 'special-token' : ''}" 
                data-token="${token}" 
                data-idx="${idx}"
                type="button"
                title="${hasHotspot ? 'Strong Cross-Attention Focus' : 'Standard Context Token'}"
              >
                <span class="token-idx">${idx}</span>
                <span class="token-text">${escapeHtml(token)}</span>
                ${hasHotspot ? '<span class="token-glow-dot"></span>' : ''}
              </button>
            `;
          }).join('')}
        </div>
        
        <!-- Token Embedding Vector Preview -->
        <div class="token-embedding-preview" id="token-embedding-preview">
          ${this.renderEmbeddingSnippet(this.selectedToken)}
        </div>
      </div>
    `;

    // Bind token chip click events
    this.container.querySelectorAll('.token-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const token = chip.dataset.token;
        this.selectedToken = token;
        this.container.querySelectorAll('.token-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');

        const preview = document.getElementById('token-embedding-preview');
        if (preview) {
          preview.innerHTML = this.renderEmbeddingSnippet(token);
        }

        const hotspotKey = this.findHotspotKeyForToken(token);
        this.onTokenSelect({
          token,
          index: parseInt(chip.dataset.idx, 10),
          hotspot: hotspotKey ? this.hotspots[hotspotKey] : null,
          hasAttention: !!hotspotKey
        });
      });
    });
  }

  hasHotspotForToken(token) {
    return !!this.findHotspotKeyForToken(token);
  }

  findHotspotKeyForToken(token) {
    if (!token) return null;
    const clean = token.toLowerCase().replace(/[^a-z0-9]/g, '');
    for (const key of Object.keys(this.hotspots)) {
      const cleanKey = key.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (cleanKey.includes(clean) || clean.includes(cleanKey)) {
        return key;
      }
    }
    return null;
  }

  renderEmbeddingSnippet(token) {
    if (!token) return '<div class="emb-hint">Select a token to inspect its 768-dimensional text embedding vector.</div>';

    // Synthesize deterministic 768-D embedding snippet for educational view
    const seed = hashString(token);
    const dims = [];
    for (let i = 0; i < 8; i++) {
      const val = (Math.sin(seed + i * 1.7) * 0.85).toFixed(4);
      dims.push(val);
    }

    return `
      <div class="embedding-info-card">
        <div class="emb-meta">
          <span class="emb-tag">CLIP ViT-L/14 Embedding</span>
          <span class="emb-dim">Dimension: 1 × 768 (Float32)</span>
        </div>
        <div class="emb-vector-row">
          <span class="emb-token-badge">"${escapeHtml(token)}"</span>
          <span class="emb-arrow">➔</span>
          <span class="emb-vector">[${dims.join(', ')}, … +760 dims]</span>
        </div>
      </div>
    `;
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

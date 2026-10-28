# Diffusion Explainer: Visualizing Stable Diffusion Step-by-Step

An interactive, browser-based visual explanation of **Stable Diffusion**, inspired by the seminal work of [Polo Club of Data Science at Georgia Tech](https://poloclub.github.io/diffusion-explainer/).

Designed to be hosted with zero backend directly via **GitHub Pages**.

---

## 🌟 Live Demo on GitHub Pages
> **Live Site:** `https://martianunlimited.github.io/diffusion-explainer/`

---

## 💡 What is Diffusion Explainer?

Diffusion Explainer breaks down the mathematical and architectural mechanics of **Stable Diffusion** into intuitive, interactive visual modules. Without requiring high-end GPUs, command lines, or deep learning libraries, anyone can explore how text prompts transform into high-resolution images.

### The 3 Core Stages of Stable Diffusion
1. **Stage 1: Text Representation (CLIP Text Encoder)**
   - Parses the text prompt into discrete tokens (up to 77 tokens).
   - Maps each token into a 768-dimensional semantic embedding vector ($\mathbf{c} \in \mathbb{R}^{77 \times 768}$).
2. **Stage 2: Iterative Latent Denoising (U-Net & Scheduler)**
   - Starts with pure Gaussian noise $\mathbf{z}_{50} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ in the compact **Latent Space** ($64 \times 64 \times 4$).
   - Over 50 denoising timesteps ($t = 1000 \to 0$), the 860M parameter U-Net predicts the noise $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})$.
   - **Cross-Attention** layers correlate spatial visual queries ($Q$) with text prompt keys/values ($K, V$), steering where specific visual concepts coalesce.
3. **Stage 3: Image Upscaling (VAE Decoder)**
   - Once fully denoised at $t = 0$, the $64 \times 64 \times 4$ latent representation is passed through the Variational Autoencoder (VAE) decoder once.
   - Upscales the compact latent representation into a crisp $512 \times 512 \times 3$ RGB image ($48\times$ computational savings compared to pixel-space diffusion!).

---

## 🔬 Interactive Features

### 1. Interactive Timestep Scrubber ($t = 50 \to 0$)
- Fluid slider allowing bidirectional scrubbing from pure random noise ($t=50$) through rough silhouettes ($t=35$), clear structures ($t=20$), sharp details ($t=10$), down to final clean RGB output ($t=0$).
- Play/Pause animation engine with $0.5\times$, $1.0\times$, and $2.0\times$ speed controls.
- Keyboard shortcuts: `Space` (Play/Pause), `ArrowLeft`/`ArrowRight` (Step $\pm 1$), `Home` ($t=50$), `End` ($t=0$).

### 2. 64×64×4 Latent Space Representation
- **PCA RGB Projection:** Converts 4-channel latent vectors into a vibrant 3-channel color representation using the standard Stable Diffusion PCA projection matrix.
- **Individual Channel Inspector:** View individual latent dimensions:
  - **Channel 0 ($z_0$):** Low-frequency luminance and global brightness.
  - **Channel 1 ($z_1$):** Pink/Green chromatic axis.
  - **Channel 2 ($z_2$):** Blue/Yellow chromatic axis.
  - **Channel 3 ($z_3$):** High-frequency fine textures and edge transitions.
- **2×2 Channel Matrix:** View all 4 latent channels side-by-side.
- **Colormaps:** Switch between *Turbo*, *Magma*, *Viridis*, *Inferno*, *Electric Cyan*, and *Grayscale*.
- **Live Tensor Statistics:** Mean ($\mu$), Standard Deviation ($\sigma$), Min/Max, and real-time hover inspection showing exact $(x, y, z_0, z_1, z_2, z_3)$ coordinates.

### 3. Cross-Attention Spatial Heatmap Explorer
- Click any prompt token (e.g., `"astronaut"`, `"helmet"`, `"red panda"`, `"waterfalls"`, `"neon"`, `"mountain"`) to inspect the spatial Cross-Attention Heatmap overlaid on the image and latents.
- Shows centroid coordinates, attention density, and spatial receptive coverage.
- Adjustable overlay alpha opacity slider.

### 4. Interactive U-Net Architecture Diagram (860M Parameters)
- Interactive SVG diagram illustrating the U-Net backbone:
  - Downsampling Blocks (ResNet + Cross-Attention)
  - Middle Bottleneck Block (ResNet + Self-Attention + Cross-Attention)
  - Upsampling Blocks + Skip Connections
  - Timestep Conditioning ($\mathbf{t}$-embedding) and Text Conditioning (CLIP $\mathbf{c}$)
- Click any block to view its tensor shapes, input/output channels, and architectural role.

### 5. Classifier-Free Guidance (CFG) Explorer
- Interactive guidance scale slider ($s \in [1.0, 20.0]$).
- 2D vector graphic demonstrating:
  $$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\text{uncond}} + s \cdot (\boldsymbol{\epsilon}_{\text{cond}} - \boldsymbol{\epsilon}_{\text{uncond}})$$
- Live assessments of under-conditioned ($s \le 3$), optimal ($s = 7\text{–}8.5$), and over-saturated ($s \ge 15$) guidance regimes.

### 6. Noise Scheduler Curves (Signal vs. Noise)
- Real-time SVG chart plotting Signal retention ($\bar{\alpha}_t$), Noise retention ($\sigma_t$), and Signal-to-Noise Ratio (SNR in dB).
- Synchronized tracking cursor and click-to-seek functionality.

### 7. Latent vs. Pixel Diffusion Comparison
- Comparative metrics explaining why Latent Diffusion achieves $48\times$ compression efficiency over older pixel-space diffusion models (e.g. DDPM, Imagen).

---

## 🚀 Running Locally

This project requires **zero build tools** and **zero dependencies**. It runs in any modern web browser via a simple HTTP server:

```bash
# Clone the repository
git clone https://github.com/martianunlimited/diffusion-explainer.git
cd diffusion-explainer

# Start a local static server
python -m http.server 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🌐 Deploying to GitHub Pages (1-Click)

1. Push this repository to GitHub:
   ```bash
   git remote add origin https://github.com/martianunlimited/diffusion-explainer.git
   git push -u origin main
   ```
2. In your GitHub repository:
   - Go to **Settings** $\to$ **Pages**
   - Under **Build and deployment**:
     - **Source:** `Deploy from a branch`
     - **Branch:** `main` / `/(root)`
   - Click **Save**.
3. Within 1–2 minutes, your site will be live at:
   `https://martianunlimited.github.io/diffusion-explainer/`

---

## 📚 Acknowledgements & References

- **Diffusion Explainer:** Wang et al., *"Diffusion Explainer: Visual Explanation for Text-to-image Stable Diffusion"*, IEEE VIS. [Polo Club at Georgia Tech](https://poloclub.github.io/diffusion-explainer/).
- **Stable Diffusion:** Rombach et al., *"High-Resolution Image Synthesis with Latent Diffusion Models"*, CVPR 2022.
- **CLIP:** Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, OpenAI 2021.
- **Denoising Diffusion Probabilistic Models (DDPM):** Ho et al., NeurIPS 2020.

---

## 📄 License
MIT License • Created by [Martian Unlimited](https://github.com/martianunlimited).

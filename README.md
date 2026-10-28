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

## 🔬 Dual Interactive Modes

The application features two complete, seamlessly switchable modes:

### Mode 1: 📖 Interactive Explainer
Explore the inner mechanics of Stable Diffusion step-by-step through pre-computed multi-step trajectories across diverse prompts:
- **Interactive Timestep Scrubber ($t = 50 \to 0$):** Fluid bidirectional scrubbing, play/pause animation ($0.5\times, 1.0\times, 2.0\times$), milestone jumps.
- **64×64×4 Latent Space Inspector:** PCA RGB projection, individual channel views ($z_0 \dots z_3$), $2\times 2$ grid matrix, colormaps, tensor statistics, and hover $(x, y)$ coordinate inspection.
- **Cross-Attention Heatmaps:** Click any prompt token to overlay its spatial attention heatmap onto the evolving image.
- **U-Net Architecture Diagram (860M Params):** Interactive SVG diagram with clickable blocks and skip connections.
- **Classifier-Free Guidance (CFG) Vector Explorer:** Interactive vector math diagram demonstrating conditional vs. unconditional noise scaling.
- **Noise Scheduler Curves:** SVG plot of Signal ($\bar{\alpha}_t$), Noise ($\sigma_t$), and SNR (dB).
- **Latent vs. Pixel Diffusion Comparison:** Detailed breakdown of the $48\times$ computational compression.

---

### Mode 2: ⚡ Live WebGPU Studio (Run Custom Prompts in Browser)
Generate images from **your own custom prompts** directly on your local GPU hardware via WebGPU:
- **🧠 Real WebGPU Neural Network (Stability AI SD-Turbo ONNX):** Runs genuine deep learning model weights (~2.4 GB) directly inside the browser using ONNX Runtime Web and WebGPU. Model files (CLIP text encoder, 860M-parameter U-Net, and VAE decoder) are downloaded from Hugging Face and cached locally into your browser's **IndexedDB** for fast subsequent loads.
- **🔗 Local / Cloud Server API Connection:** Connect directly to a local GPU rig (Automatic1111/SD.Next at `http://127.0.0.1:7860` or ComfyUI at `http://127.0.0.1:8188`) or Hugging Face Serverless Inference API for zero-download generation.
- **📐 Educational Latent Math Simulator:** Lightweight 50ms mathematical visualization of reverse diffusion ODE/SDE curves and PCA latent space projections for instant exploration without downloading gigabytes of weights.
- **Interactive Hyperparameter Tuning:**
  - **Inference Steps:** 1–4 steps (SD-Turbo distilled) up to 25 steps.
  - **Guidance Scale (CFG):** 0.0 to 15.0 slider.
  - **Seed Control:** Numeric input + randomize dice button 🎲.
  - **Resolution:** $512\times 512$ (Standard SD) or $256\times 256$ preview.
- **Session History Gallery:** Automatically saves every generated image to a session strip with 1-click PNG download, prompt copying, and "Send to Explainer" transfer.

---

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

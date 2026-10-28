/**
 * Direct Connection to Local / Remote Diffusion Backends
 * Connects directly to:
 * - Local Automatic1111 / SD.Next WebUI (http://127.0.0.1:7860)
 * - Local ComfyUI (http://127.0.0.1:8188)
 * - Hugging Face Free Serverless Inference API
 */

export class ServerBackendEngine {
  constructor() {
    this.backendType = 'auto1111'; // 'auto1111' | 'hf-api'
    this.endpointUrl = 'http://127.0.0.1:7860';
    this.hfToken = '';
    this.hfModel = 'stabilityai/stable-diffusion-v1-5';
  }

  setBackend(type, url, token, model) {
    this.backendType = type;
    if (url) this.endpointUrl = url;
    if (token) this.hfToken = token;
    if (model) this.hfModel = model;
  }

  async generate({ prompt, negativePrompt = '', steps = 20, guidanceScale = 7.5, seed = -1, width = 512, height = 512 }) {
    const startTime = performance.now();

    if (this.backendType === 'auto1111') {
      const url = `${this.endpointUrl.replace(/\/$/, '')}/sdapi/v1/txt2img`;
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePrompt,
          steps,
          cfg_scale: guidanceScale,
          seed,
          width,
          height
        })
      });

      if (!response.ok) {
        throw new Error(`Automatic1111 returned HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      if (!data.images || data.images.length === 0) {
        throw new Error('No image returned by Automatic1111 API');
      }

      const canvas = await this.base64ToCanvas(data.images[0]);
      return {
        canvas,
        prompt,
        seed,
        steps,
        totalTimeMs: performance.now() - startTime,
        backend: 'local-auto1111'
      };
    } else if (this.backendType === 'hf-api') {
      if (!this.hfToken) {
        throw new Error('Hugging Face API token is required. Enter your token (hf_...) in settings.');
      }

      const url = `https://api-inference.huggingface.co/models/${this.hfModel}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.hfToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          inputs: prompt,
          parameters: {
            negative_prompt: negativePrompt,
            num_inference_steps: steps,
            guidance_scale: guidanceScale,
            seed: seed > 0 ? seed : undefined
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Hugging Face API returned HTTP ${response.status}: ${errorText}`);
      }

      const blob = await response.blob();
      const canvas = await this.blobToCanvas(blob);
      return {
        canvas,
        prompt,
        seed,
        steps,
        totalTimeMs: performance.now() - startTime,
        backend: 'huggingface-api'
      };
    }

    throw new Error(`Unsupported backend type: ${this.backendType}`);
  }

  base64ToCanvas(base64) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        resolve(canvas);
      };
      img.onerror = () => reject(new Error('Failed to load image from API response'));
      img.src = `data:image/png;base64,${base64}`;
    });
  }

  blobToCanvas(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        resolve(canvas);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load image blob from API response'));
      };
      img.src = url;
    });
  }
}

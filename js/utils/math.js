/**
 * Mathematics and Tensor Simulation Utilities for Stable Diffusion
 * Includes:
 * - 4-channel Latent PCA-to-RGB projection
 * - Gaussian noise and latent forward diffusion synthesis
 * - DDIM/Euler noise schedule curves (alpha, sigma, SNR)
 * - 2D Cross-Attention Gaussian hotspot generator
 */

// PCA projection matrix to convert 4-channel SD latents to 3-channel RGB (diffusers standard)
export const LATENT_RGB_PROJECTION = [
  [0.298, 0.207, -0.064, 0.111],  // Red
  [0.207, 0.286, 0.183, -0.133],  // Green
  [-0.064, 0.183, 0.602, 0.076]   // Blue
];

/**
 * Generates pseudorandom standard normal numbers using Box-Muller transform
 */
export function seededRandomGaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Simple Mulberry32 seeded pseudo-random number generator
 */
export function createRNG(seed) {
  let s = Math.abs(seed) || 12345;
  return function() {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Computes noise schedule parameters for given step (0..50)
 */
export function getScheduleParams(step, totalSteps = 50) {
  const tRatio = step / totalSteps; // 1.0 (step 50, pure noise) down to 0.0 (step 0, clean)
  // Cosine-inspired schedule
  const alphaBar = Math.max(0.001, Math.min(1.0, 1.0 - Math.pow(tRatio, 1.55)));
  const sigma = Math.sqrt(Math.max(0.0, 1.0 - alphaBar));
  const snr = alphaBar / Math.max(1e-5, 1.0 - alphaBar);
  const snrDb = 10 * Math.log10(snr);

  return {
    step,
    timestep: Math.round(step * (1000 / totalSteps)),
    alphaBar,
    sigma,
    snr,
    snrDb
  };
}

/**
 * Projects a 4-channel latent vector [z0, z1, z2, z3] to [r, g, b] in [0, 255]
 */
export function projectLatentToRGB(z0, z1, z2, z3) {
  const rNorm = LATENT_RGB_PROJECTION[0][0] * z0 + LATENT_RGB_PROJECTION[0][1] * z1 + LATENT_RGB_PROJECTION[0][2] * z2 + LATENT_RGB_PROJECTION[0][3] * z3;
  const gNorm = LATENT_RGB_PROJECTION[1][0] * z0 + LATENT_RGB_PROJECTION[1][1] * z1 + LATENT_RGB_PROJECTION[1][2] * z2 + LATENT_RGB_PROJECTION[1][3] * z3;
  const bNorm = LATENT_RGB_PROJECTION[2][0] * z0 + LATENT_RGB_PROJECTION[2][1] * z1 + LATENT_RGB_PROJECTION[2][2] * z2 + LATENT_RGB_PROJECTION[2][3] * z3;

  // Center around 0.5 and scale to uint8
  const r = Math.round(Math.max(0, Math.min(1, rNorm + 0.5)) * 255);
  const g = Math.round(Math.max(0, Math.min(1, gNorm + 0.5)) * 255);
  const b = Math.round(Math.max(0, Math.min(1, bNorm + 0.5)) * 255);

  return [r, g, b];
}

/**
 * Synthesizes a 64x64x4 latent tensor from an input 64x64 downsampled image
 * and blends it with noise based on step (0..50)
 */
export function generateLatentTensor(clean64ImageData, step, totalSteps = 50, seed = 42) {
  const width = 64;
  const height = 64;
  const planeSize = width * height;
  const params = getScheduleParams(step, totalSteps);
  const signalWeight = Math.sqrt(params.alphaBar);
  const noiseWeight = Math.sqrt(1.0 - params.alphaBar);

  const rng = createRNG(seed + step * 100);
  const latents = new Float32Array(4 * planeSize);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const imgIdx = idx * 4;

      const r = clean64ImageData ? clean64ImageData[imgIdx] / 255.0 - 0.5 : 0;
      const g = clean64ImageData ? clean64ImageData[imgIdx + 1] / 255.0 - 0.5 : 0;
      const b = clean64ImageData ? clean64ImageData[imgIdx + 2] / 255.0 - 0.5 : 0;

      // Inverse approximation of latent channels from clean RGB:
      // z0: Luminance / contrast
      const targetZ0 = 0.299 * r + 0.587 * g + 0.114 * b;
      // z1: Red/Cyan chroma axis
      const targetZ1 = r - g;
      // z2: Blue/Yellow chroma axis
      const targetZ2 = b - (r + g) * 0.5;
      // z3: High frequency texture details
      const targetZ3 = (r * g - b * b) * 1.5;

      // Add Gaussian noise
      const n0 = seededRandomGaussian(rng);
      const n1 = seededRandomGaussian(rng);
      const n2 = seededRandomGaussian(rng);
      const n3 = seededRandomGaussian(rng);

      latents[idx] = signalWeight * targetZ0 + noiseWeight * n0;
      latents[planeSize + idx] = signalWeight * targetZ1 + noiseWeight * n1;
      latents[2 * planeSize + idx] = signalWeight * targetZ2 + noiseWeight * n2;
      latents[3 * planeSize + idx] = signalWeight * targetZ3 + noiseWeight * n3;
    }
  }

  return {
    width,
    height,
    channels: 4,
    data: latents,
    params
  };
}

/**
 * Computes a 64x64 Cross-Attention Spatial Heatmap for a given token hotspot
 */
export function computeAttentionHeatmap(hotspot, width = 64, height = 64) {
  const map = new Float32Array(width * height);
  if (!hotspot) {
    // Uniform baseline attention
    map.fill(0.1);
    return map;
  }

  const { cx, cy, rx, ry, intensity = 1.0 } = hotspot;
  const sigmaX = rx || 0.3;
  const sigmaY = ry || 0.3;

  let maxVal = 0.0;
  for (let y = 0; y < height; y++) {
    const ny = y / height;
    const dy = (ny - cy) / sigmaY;

    for (let x = 0; x < width; x++) {
      const nx = x / width;
      const dx = (nx - cx) / sigmaX;

      // 2D Gaussian density with subtle background noise
      const distSq = dx * dx + dy * dy;
      const val = Math.exp(-0.5 * distSq) * intensity + 0.04;
      map[y * width + x] = val;
      if (val > maxVal) maxVal = val;
    }
  }

  // Normalize to [0, 1]
  if (maxVal > 0) {
    for (let i = 0; i < map.length; i++) {
      map[i] = Math.min(1.0, map[i] / maxVal);
    }
  }

  return map;
}

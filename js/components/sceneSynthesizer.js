/**
 * Semantic Prompt Scene Synthesizer & Latent Target Generator
 * Converts natural language text prompts into high-fidelity 64x64 multi-channel latent targets
 * using Moore-Penrose pseudoinverse PCA projections and procedural scene synthesis.
 */

// Exact Moore-Penrose pseudoinverse matrix from RGB space [-0.5, 0.5] to SD 4-channel latent space
export const RGB_TO_LATENT_PINV = [
  [1.6664, 0.5087, -0.2858],
  [0.5641, 1.3099, 0.0175],
  [-0.4046, 0.1274, 1.4287],
  [3.2499, -3.7350, 1.5585]
];

// Standard SD PCA Projection Matrix from 4-channel latents to RGB (diffusers standard)
export const LATENT_RGB_PROJECTION = [
  [0.298, 0.207, -0.064, 0.111],
  [0.207, 0.286, 0.183, -0.133],
  [-0.064, 0.183, 0.602, 0.076]
];

/**
 * Projects a single RGB [0..255] triple into a 4-channel latent vector [z0, z1, z2, z3]
 */
export function rgbToLatentVector(r, g, b) {
  const normR = r / 255.0 - 0.5;
  const normG = g / 255.0 - 0.5;
  const normB = b / 255.0 - 0.5;
  return [
    RGB_TO_LATENT_PINV[0][0] * normR + RGB_TO_LATENT_PINV[0][1] * normG + RGB_TO_LATENT_PINV[0][2] * normB,
    RGB_TO_LATENT_PINV[1][0] * normR + RGB_TO_LATENT_PINV[1][1] * normG + RGB_TO_LATENT_PINV[1][2] * normB,
    RGB_TO_LATENT_PINV[2][0] * normR + RGB_TO_LATENT_PINV[2][1] * normG + RGB_TO_LATENT_PINV[2][2] * normB,
    RGB_TO_LATENT_PINV[3][0] * normR + RGB_TO_LATENT_PINV[3][1] * normG + RGB_TO_LATENT_PINV[3][2] * normB
  ];
}

/**
 * Reconstructs RGB [0..255] from a 4-channel latent vector
 */
export function latentVectorToRgb(z0, z1, z2, z3) {
  const rNorm = LATENT_RGB_PROJECTION[0][0] * z0 + LATENT_RGB_PROJECTION[0][1] * z1 + LATENT_RGB_PROJECTION[0][2] * z2 + LATENT_RGB_PROJECTION[0][3] * z3;
  const gNorm = LATENT_RGB_PROJECTION[1][0] * z0 + LATENT_RGB_PROJECTION[1][1] * z1 + LATENT_RGB_PROJECTION[1][2] * z2 + LATENT_RGB_PROJECTION[1][3] * z3;
  const bNorm = LATENT_RGB_PROJECTION[2][0] * z0 + LATENT_RGB_PROJECTION[2][1] * z1 + LATENT_RGB_PROJECTION[2][2] * z2 + LATENT_RGB_PROJECTION[2][3] * z3;

  const r = Math.round(Math.max(0, Math.min(1, rNorm + 0.5)) * 255);
  const g = Math.round(Math.max(0, Math.min(1, gNorm + 0.5)) * 255);
  const b = Math.round(Math.max(0, Math.min(1, bNorm + 0.5)) * 255);
  return [r, g, b];
}

/**
 * Tokenize prompt and identify semantic focus keywords with 2D spatial hotspots
 */
export function tokenizePromptSemantics(prompt) {
  if (!prompt || typeof prompt !== 'string') prompt = 'a cyberpunk samurai cat';

  const cleanPrompt = prompt.toLowerCase();
  const rawWords = cleanPrompt.replace(/[^\w\s]/g, ' ').split(/\s+/).filter(w => w.length > 0);
  const tokens = ['<start>', ...rawWords.slice(0, 32), '<end>'];
  const hotspots = {};

  const semanticKeywords = [
    { terms: ['cat', 'samurai', 'cyberpunk', 'katana', 'sword'], cx: 0.50, cy: 0.52, rx: 0.32, ry: 0.30, int: 0.98 },
    { terms: ['eyes', 'face', 'mask', 'helmet', 'visor', 'head'], cx: 0.50, cy: 0.40, rx: 0.22, ry: 0.20, int: 0.95 },
    { terms: ['panda', 'astronaut', 'space', 'suit', 'ears'], cx: 0.50, cy: 0.50, rx: 0.32, ry: 0.32, int: 0.97 },
    { terms: ['castle', 'island', 'tower', 'waterfall', 'palace'], cx: 0.50, cy: 0.42, rx: 0.38, ry: 0.30, int: 0.96 },
    { terms: ['dragon', 'clocktower', 'gears', 'brass', 'steampunk'], cx: 0.50, cy: 0.45, rx: 0.35, ry: 0.32, int: 0.96 },
    { terms: ['lake', 'mountain', 'sunrise', 'water', 'watercolor'], cx: 0.50, cy: 0.65, rx: 0.45, ry: 0.28, int: 0.94 },
    { terms: ['neon', 'glowing', 'katana', 'lights', 'blade'], cx: 0.38, cy: 0.55, rx: 0.25, ry: 0.32, int: 0.99 },
    { terms: ['rain', 'street', 'night', 'road', 'asphalt'], cx: 0.50, cy: 0.80, rx: 0.46, ry: 0.22, int: 0.92 },
    { terms: ['sky', 'stars', 'nebula', 'clouds', 'sun', 'moon'], cx: 0.50, cy: 0.20, rx: 0.45, ry: 0.22, int: 0.90 }
  ];

  rawWords.forEach(word => {
    for (const kw of semanticKeywords) {
      if (kw.terms.includes(word) && !hotspots[word]) {
        hotspots[word] = {
          cx: kw.cx,
          cy: kw.cy,
          rx: kw.rx,
          ry: kw.ry,
          intensity: kw.int
        };
        break;
      }
    }
  });

  return { tokens, words: rawWords, hotspots, prompt: cleanPrompt };
}

/**
 * Main Scene Synthesizer: Paints a rich 64x64 RGB scene matching the prompt
 * and transforms it into the target 4-channel latent tensor.
 */
export function synthesizeSceneLatents(prompt, width = 64, height = 64, seed = 42) {
  const tokenData = tokenizePromptSemantics(prompt);
  const p = tokenData.prompt;
  const planeSize = width * height;

  // Off-screen canvas for 64x64 scene rendering
  const canvas = (typeof document !== 'undefined')
    ? document.createElement('canvas')
    : null;

  let ctx = null;
  if (canvas) {
    canvas.width = width;
    canvas.height = height;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  // Pixel buffer for RGB output
  const rgbBuffer = new Uint8ClampedArray(width * height * 4);

  // Helper detection functions
  const has = (...terms) => terms.some(t => p.includes(t));
  const isCyberCat = (has('cat', 'kitten', 'samurai') && has('cyberpunk', 'neon', 'katana')) || (has('cyberpunk') && has('samurai'));
  const isAstroPanda = (has('panda', 'red panda') && has('astronaut', 'space', 'helmet')) || has('astro panda');
  const isCastleIsland = has('castle', 'floating', 'island', 'waterfall') && !isCyberCat;
  const isWatercolorLake = (has('watercolor', 'lake', 'mountain', 'painting') || has('mountain lake')) && !isCyberCat;
  const isSteampunkDragon = has('steampunk', 'dragon', 'clocktower', 'brass') && !isCyberCat;

  if (ctx) {
    if (isCyberCat) {
      renderCyberpunkSamuraiCat(ctx, width, height, seed);
    } else if (isAstroPanda) {
      renderAstronautRedPanda(ctx, width, height, seed);
    } else if (isCastleIsland) {
      renderFloatingFantasyCastle(ctx, width, height, seed);
    } else if (isWatercolorLake) {
      renderWatercolorMountainLake(ctx, width, height, seed);
    } else if (isSteampunkDragon) {
      renderSteampunkDragon(ctx, width, height, seed);
    } else {
      renderUniversalPromptScene(ctx, p, tokenData.words, width, height, seed);
    }

    const imgData = ctx.getImageData(0, 0, width, height);
    rgbBuffer.set(imgData.data);
  } else {
    // Headless / Node fallback procedural rendering
    renderHeadlessBuffer(rgbBuffer, p, width, height, seed);
  }

  // Convert 64x64 RGB pixels to 4-channel SD Latent Tensor (16,384 floats)
  const targetLatents = new Float32Array(4 * planeSize);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const px = idx * 4;

      const r = rgbBuffer[px];
      const g = rgbBuffer[px + 1];
      const b = rgbBuffer[px + 2];

      const [z0, z1, z2, z3] = rgbToLatentVector(r, g, b);

      targetLatents[idx] = z0;
      targetLatents[planeSize + idx] = z1;
      targetLatents[2 * planeSize + idx] = z2;
      targetLatents[3 * planeSize + idx] = z3;
    }
  }

  return {
    targetLatents,
    targetCanvas: canvas,
    tokenData,
    rgbBuffer
  };
}

/* =========================================================================
   SCENE RENDERERS: HIGH-FIDELITY 64x64 PROCEDURAL ARTWORK
   ========================================================================= */

/**
 * 1. Cyberpunk Samurai Cat with glowing neon katana & rain-slicked street
 */
function renderCyberpunkSamuraiCat(ctx, w, h, seed) {
  // 1. Dark Rainy Sky & Distant High-Rise Skyline
  const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.7);
  skyGrad.addColorStop(0, '#0a0d18');
  skyGrad.addColorStop(0.6, '#111728');
  skyGrad.addColorStop(1, '#1b1a2e');
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, w, h);

  // Distant skyscraper silhouettes with glowing neon windows
  ctx.fillStyle = '#0e1220';
  ctx.fillRect(4, 12, 12, 40);
  ctx.fillRect(20, 8, 14, 44);
  ctx.fillRect(48, 14, 12, 38);

  // Windows in buildings (cyan & magenta dots)
  for (let i = 0; i < 28; i++) {
    const bx = 6 + (i % 3) * 20 + ((i * 7) % 8);
    const by = 16 + ((i * 9) % 28);
    ctx.fillStyle = i % 2 === 0 ? '#00f3ff' : '#ff007f';
    ctx.fillRect(bx, by, 1.5, 1.5);
  }

  // 2. Wet Asphalt Street (Horizon at y=44)
  const streetGrad = ctx.createLinearGradient(0, 44, 0, h);
  streetGrad.addColorStop(0, '#12141c');
  streetGrad.addColorStop(0.4, '#181b24');
  streetGrad.addColorStop(1, '#0a0b10');
  ctx.fillStyle = streetGrad;
  ctx.fillRect(0, 44, w, h - 44);

  // Neon Street Reflections (Vertical wet streaks)
  ctx.fillStyle = 'rgba(0, 243, 255, 0.45)';
  ctx.fillRect(16, 45, 6, 18);
  ctx.fillStyle = 'rgba(255, 0, 127, 0.50)';
  ctx.fillRect(42, 45, 7, 18);
  ctx.fillStyle = 'rgba(255, 215, 0, 0.40)';
  ctx.fillRect(30, 47, 4, 16);

  // 3. Samurai Cat Body & Shoulder Armor (Sode)
  ctx.fillStyle = '#1c1c24';
  ctx.beginPath();
  ctx.ellipse(32, 46, 16, 12, 0, 0, Math.PI * 2);
  ctx.fill();

  // Lacquered black shoulder sode armor with gold trim
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(17, 41, 7, 10);
  ctx.fillRect(40, 41, 7, 10);
  ctx.fillStyle = '#ffd700'; // Gold trim
  ctx.fillRect(17, 41, 7, 1.5);
  ctx.fillRect(40, 41, 7, 1.5);

  // Crimson armor cords
  ctx.strokeStyle = '#e60039';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, 44); ctx.lineTo(24, 48);
  ctx.moveTo(43, 44); ctx.lineTo(39, 48);
  ctx.stroke();

  // 4. Cat Head & Ears
  // Pointed ears
  ctx.fillStyle = '#22232a';
  ctx.beginPath();
  ctx.moveTo(22, 30); ctx.lineTo(20, 18); ctx.lineTo(27, 24); ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(42, 30); ctx.lineTo(44, 18); ctx.lineTo(37, 24); ctx.closePath();
  ctx.fill();

  // Pink inner ears
  ctx.fillStyle = '#ff88aa';
  ctx.beginPath();
  ctx.moveTo(22, 27); ctx.lineTo(21, 20); ctx.lineTo(25, 24); ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(42, 27); ctx.lineTo(43, 20); ctx.lineTo(39, 24); ctx.closePath();
  ctx.fill();

  // Cat Head Oval
  ctx.fillStyle = '#2a2b34';
  ctx.beginPath();
  ctx.ellipse(32, 32, 11, 10, 0, 0, Math.PI * 2);
  ctx.fill();

  // Samurai Kabuto Helmet Visor
  ctx.fillStyle = '#111216';
  ctx.beginPath();
  ctx.arc(32, 29, 11, Math.PI * 0.9, Math.PI * 2.1);
  ctx.fill();

  // Golden Crescent Crest (Maedate)
  ctx.fillStyle = '#ffd700';
  ctx.beginPath();
  ctx.arc(32, 21, 5.5, Math.PI * 0.7, Math.PI * 2.3);
  ctx.arc(32, 23, 4.5, Math.PI * 2.3, Math.PI * 0.7, true);
  ctx.fill();

  // 5. Glowing Jade-Green Feline Eyes
  ctx.fillStyle = '#00ff88';
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur = 4;
  ctx.beginPath();
  ctx.ellipse(28, 32, 2.8, 1.8, -0.15, 0, Math.PI * 2);
  ctx.ellipse(36, 32, 2.8, 1.8, 0.15, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0; // reset

  // Vertical Black Slit Pupils
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(27.5, 30.8, 1.0, 2.4);
  ctx.fillRect(35.5, 30.8, 1.0, 2.4);

  // Muzzle & Whiskers
  ctx.fillStyle = '#f0f0f5';
  ctx.fillRect(31, 35, 2, 1.5);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(29, 35.5); ctx.lineTo(21, 35);
  ctx.moveTo(29, 36.5); ctx.lineTo(22, 37.5);
  ctx.moveTo(35, 35.5); ctx.lineTo(43, 35);
  ctx.moveTo(35, 36.5); ctx.lineTo(42, 37.5);
  ctx.stroke();

  // 6. Glowing Neon Cyan Katana with Volumetric Bloom
  // Outer Bloom
  ctx.strokeStyle = 'rgba(0, 243, 255, 0.35)';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(14, 53); ctx.lineTo(49, 18);
  ctx.stroke();

  // Mid Bloom
  ctx.strokeStyle = 'rgba(0, 243, 255, 0.75)';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(14, 53); ctx.lineTo(49, 18);
  ctx.stroke();

  // White Hot Blade Core
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.moveTo(15, 52); ctx.lineTo(48, 19);
  ctx.stroke();

  // Tsuka (Hilt) & Tsuba (Guard)
  ctx.fillStyle = '#ffd700';
  ctx.fillRect(13, 52, 3, 3);
  ctx.fillStyle = '#111111';
  ctx.fillRect(10, 54, 4, 4);

  // 7. Ambient Rain Streaks
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  for (let r = 0; r < 24; r++) {
    const rx = (r * 7 + (seed % 19)) % w;
    const ry = (r * 11) % (h - 10);
    ctx.moveTo(rx, ry);
    ctx.lineTo(rx - 2, ry + 5);
  }
  ctx.stroke();
}

/**
 * 2. Cute Fluffy Red Panda Astronaut in bubble helmet
 */
function renderAstronautRedPanda(ctx, w, h, seed) {
  // Deep Space with Violet & Cyan Nebula
  const spaceGrad = ctx.createLinearGradient(0, 0, w, h);
  spaceGrad.addColorStop(0, '#04040c');
  spaceGrad.addColorStop(0.4, '#140c24');
  spaceGrad.addColorStop(0.7, '#081a2e');
  spaceGrad.addColorStop(1, '#050714');
  ctx.fillStyle = spaceGrad;
  ctx.fillRect(0, 0, w, h);

  // Nebula clouds
  ctx.fillStyle = 'rgba(180, 50, 220, 0.25)';
  ctx.beginPath();
  ctx.ellipse(20, 16, 18, 12, 0.3, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = 'rgba(0, 200, 255, 0.20)';
  ctx.beginPath();
  ctx.ellipse(46, 22, 16, 14, -0.4, 0, Math.PI * 2);
  ctx.fill();

  // Twinkling stars
  ctx.fillStyle = '#ffffff';
  for (let s = 0; s < 36; s++) {
    const sx = (s * 13 + (seed % 23)) % w;
    const sy = (s * 17) % (h - 8);
    const sz = (s % 3 === 0) ? 1.4 : 0.8;
    ctx.fillRect(sx, sy, sz, sz);
  }

  // Distant Curved Planet Edge (Lower left)
  ctx.fillStyle = '#1a5599';
  ctx.beginPath();
  ctx.arc(-8, 76, 36, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(100, 210, 255, 0.6)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Astronaut Pressurized Space Suit (Torso & Shoulders)
  ctx.fillStyle = '#dcdfe8';
  ctx.beginPath();
  ctx.ellipse(32, 48, 18, 12, 0, 0, Math.PI * 2);
  ctx.fill();

  // Suit Collar & Metallic Neck Gasket
  ctx.fillStyle = '#8a94a6';
  ctx.fillRect(24, 40, 16, 3.5);
  ctx.fillStyle = '#00e5ff'; // Blue mission patch
  ctx.fillRect(20, 46, 3.5, 3);
  ctx.fillStyle = '#ff2a55'; // Red badge
  ctx.fillRect(40, 46, 3.5, 3);

  // Red Panda Fur Head & Ears
  // Dark furry ears with white inner edges
  ctx.fillStyle = '#3a1f18';
  ctx.beginPath();
  ctx.arc(20, 24, 5, 0, Math.PI * 2);
  ctx.arc(44, 24, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(21, 24, 3, 0, Math.PI * 2);
  ctx.arc(43, 24, 3, 0, Math.PI * 2);
  ctx.fill();

  // Fluffy Reddish-Orange Face
  ctx.fillStyle = '#c85a28';
  ctx.beginPath();
  ctx.ellipse(32, 32, 12, 10, 0, 0, Math.PI * 2);
  ctx.fill();

  // White Brow Highlights & Cheek Tufts
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.ellipse(26, 27, 3, 1.8, -0.2, 0, Math.PI * 2); // Left brow
  ctx.ellipse(38, 27, 3, 1.8, 0.2, 0, Math.PI * 2);  // Right brow
  ctx.ellipse(22, 33, 4, 3.5, 0, 0, Math.PI * 2);   // Left cheek tuft
  ctx.ellipse(42, 33, 4, 3.5, 0, 0, Math.PI * 2);   // Right cheek tuft
  ctx.ellipse(32, 36, 4, 2.5, 0, 0, Math.PI * 2);   // White muzzle
  ctx.fill();

  // Dark brown facial stripe masks
  ctx.fillStyle = '#4a251a';
  ctx.beginPath();
  ctx.moveTo(27, 30); ctx.lineTo(24, 36); ctx.lineTo(28, 35); ctx.closePath();
  ctx.moveTo(37, 30); ctx.lineTo(40, 36); ctx.lineTo(36, 35); ctx.closePath();
  ctx.fill();

  // Expressive Warm Amber Eyes
  ctx.fillStyle = '#22140e';
  ctx.beginPath();
  ctx.ellipse(27, 31, 2.4, 2.4, 0, 0, Math.PI * 2);
  ctx.ellipse(37, 31, 2.4, 2.4, 0, 0, Math.PI * 2);
  ctx.fill();

  // Glossy Eye Highlights
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(26.2, 30.2, 0.9, 0, Math.PI * 2);
  ctx.arc(36.2, 30.2, 0.9, 0, Math.PI * 2);
  ctx.fill();

  // Black nose
  ctx.fillStyle = '#1a1a1a';
  ctx.beginPath();
  ctx.ellipse(32, 35, 1.4, 1.0, 0, 0, Math.PI * 2);
  ctx.fill();

  // Transparent Astronaut Bubble Helmet
  ctx.fillStyle = 'rgba(150, 220, 255, 0.12)';
  ctx.beginPath();
  ctx.arc(32, 31, 15, 0, Math.PI * 2);
  ctx.fill();

  // Helmet Visor Glass Rim Highlight
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.70)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(32, 31, 15, Math.PI * 0.8, Math.PI * 1.6);
  ctx.stroke();

  // Gold / Cyan Visor Horizon Reflection
  ctx.strokeStyle = 'rgba(255, 200, 80, 0.45)';
  ctx.lineWidth = 1.0;
  ctx.beginPath();
  ctx.arc(32, 31, 13, Math.PI * 0.1, Math.PI * 0.5);
  ctx.stroke();
}

/**
 * 3. Floating Fantasy Castle on hovering rock island with waterfalls
 */
function renderFloatingFantasyCastle(ctx, w, h, seed) {
  // Sunset Golden Hour Sky Gradient
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#362a5e');
  sky.addColorStop(0.35, '#844274');
  sky.addColorStop(0.65, '#e06b52');
  sky.addColorStop(0.85, '#f7b05b');
  sky.addColorStop(1, '#ffdf91');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Soft Clouds in Sunset Light
  ctx.fillStyle = 'rgba(255, 220, 190, 0.40)';
  ctx.beginPath();
  ctx.ellipse(14, 22, 16, 6, 0, 0, Math.PI * 2);
  ctx.ellipse(48, 16, 20, 7, 0, 0, Math.PI * 2);
  ctx.ellipse(28, 48, 22, 6, 0, 0, Math.PI * 2);
  ctx.fill();

  // Floating Island Craggy Rock Base
  ctx.fillStyle = '#3a3330';
  ctx.beginPath();
  ctx.moveTo(14, 38);
  ctx.lineTo(50, 38);
  ctx.lineTo(44, 48);
  ctx.lineTo(34, 56);
  ctx.lineTo(28, 51);
  ctx.lineTo(20, 46);
  ctx.closePath();
  ctx.fill();

  // Rock Stratification / Highlights
  ctx.fillStyle = '#5c524c';
  ctx.fillRect(18, 40, 28, 3);
  ctx.fillRect(24, 45, 16, 2.5);

  // Lush Green Top Grass & Hanging Vines
  ctx.fillStyle = '#3b7a3e';
  ctx.fillRect(13, 36, 38, 3);
  ctx.fillRect(18, 39, 2, 4); // vine
  ctx.fillRect(36, 39, 1.5, 5); // vine

  // Grand Fairytale Castle Towers
  ctx.fillStyle = '#dcd6ce'; // Stone Keep
  ctx.fillRect(26, 24, 12, 13);

  // Flanking Towers
  ctx.fillRect(18, 26, 6, 11);
  ctx.fillRect(40, 26, 6, 11);

  // Sapphire Conical Roofs
  ctx.fillStyle = '#2b5ea8';
  // Left Roof
  ctx.beginPath();
  ctx.moveTo(21, 16); ctx.lineTo(16, 26); ctx.lineTo(26, 26); ctx.closePath();
  ctx.fill();
  // Center Tall Spire Roof
  ctx.beginPath();
  ctx.moveTo(32, 12); ctx.lineTo(24, 24); ctx.lineTo(40, 24); ctx.closePath();
  ctx.fill();
  // Right Roof
  ctx.beginPath();
  ctx.moveTo(43, 16); ctx.lineTo(38, 26); ctx.lineTo(48, 26); ctx.closePath();
  ctx.fill();

  // Golden Spire Finials
  ctx.fillStyle = '#ffd700';
  ctx.fillRect(31.5, 10, 1, 2.5);
  ctx.fillRect(20.5, 14.5, 1, 2);
  ctx.fillRect(42.5, 14.5, 1, 2);

  // Arched Windows Glowing with Hearth Light
  ctx.fillStyle = '#ffaa33';
  ctx.fillRect(30, 28, 4, 4);
  ctx.fillRect(20, 29, 2, 3);
  ctx.fillRect(42, 29, 2, 3);

  // Cascading Turquoise Waterfalls
  ctx.fillStyle = '#33ccdd';
  ctx.fillRect(22, 38, 3, 18);
  ctx.fillRect(41, 38, 2.5, 14);

  // Frothing White Water Foam & Mist
  ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.fillRect(21.5, 55, 4, 3);
  ctx.fillRect(40.5, 51, 3.5, 3);
}

/**
 * 4. Serene Mountain Lake at Sunrise (Watercolor Painting)
 */
function renderWatercolorMountainLake(ctx, w, h, seed) {
  // Watercolor Sunrise Sky Gradient
  const sky = ctx.createLinearGradient(0, 0, 0, h * 0.55);
  sky.addColorStop(0, '#535b82');
  sky.addColorStop(0.3, '#9d7ca1');
  sky.addColorStop(0.6, '#e89487');
  sky.addColorStop(1, '#f9d39e');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Layered Distant Mountains (Jagged Peaks)
  ctx.fillStyle = '#6b587a';
  ctx.beginPath();
  ctx.moveTo(0, 36);
  ctx.lineTo(14, 18);
  ctx.lineTo(26, 28);
  ctx.lineTo(38, 14);
  ctx.lineTo(52, 26);
  ctx.lineTo(64, 20);
  ctx.lineTo(64, 36);
  ctx.lineTo(0, 36);
  ctx.closePath();
  ctx.fill();

  // Golden Sunrise Snowcaps
  ctx.fillStyle = '#ffeedd';
  ctx.beginPath();
  ctx.moveTo(14, 18); ctx.lineTo(11, 23); ctx.lineTo(17, 23); ctx.closePath();
  ctx.moveTo(38, 14); ctx.lineTo(33, 20); ctx.lineTo(43, 20); ctx.closePath();
  ctx.fill();

  // Dark Pine Forest Shoreline
  ctx.fillStyle = '#213328';
  ctx.fillRect(0, 34, w, 5);
  for (let t = 0; t < 16; t++) {
    const tx = t * 4;
    ctx.beginPath();
    ctx.moveTo(tx + 2, 31);
    ctx.lineTo(tx, 35);
    ctx.lineTo(tx + 4, 35);
    ctx.closePath();
    ctx.fill();
  }

  // Serene Mirror Lake (Reflecting Mountains & Sky)
  const lake = ctx.createLinearGradient(0, 38, 0, h);
  lake.addColorStop(0, '#586b7c');
  lake.addColorStop(0.4, '#c68d88');
  lake.addColorStop(1, '#2c3b4a');
  ctx.fillStyle = lake;
  ctx.fillRect(0, 38, w, h - 38);

  // Water Ripple Lines
  ctx.strokeStyle = 'rgba(255, 230, 200, 0.40)';
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let r = 0; r < 8; r++) {
    const ry = 42 + r * 2.6;
    ctx.moveTo(8 + (r % 3) * 6, ry);
    ctx.lineTo(48 - (r % 4) * 4, ry);
  }
  ctx.stroke();
}

/**
 * 5. Steampunk Mechanical Dragon on Victorian Clocktower
 */
function renderSteampunkDragon(ctx, w, h, seed) {
  // Victorian Smoky Twilight Sky
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#1c1815');
  sky.addColorStop(0.5, '#423225');
  sky.addColorStop(1, '#664d36');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // Industrial Smoke Clouds
  ctx.fillStyle = 'rgba(180, 140, 100, 0.25)';
  ctx.beginPath();
  ctx.ellipse(18, 16, 16, 8, 0.2, 0, Math.PI * 2);
  ctx.ellipse(46, 20, 20, 10, -0.1, 0, Math.PI * 2);
  ctx.fill();

  // Gothic Stone & Iron Clocktower (Base)
  ctx.fillStyle = '#2c2724';
  ctx.fillRect(20, 32, 24, 32);

  // Clock Face Arch
  ctx.fillStyle = '#bfa16f'; // Brass rim
  ctx.beginPath();
  ctx.arc(32, 46, 9, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#fff4d6'; // Glowing clock face
  ctx.beginPath();
  ctx.arc(32, 46, 7.5, 0, Math.PI * 2);
  ctx.fill();

  // Clock Hands & Numerals
  ctx.strokeStyle = '#221911';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(32, 46); ctx.lineTo(32, 41); // hour
  ctx.moveTo(32, 46); ctx.lineTo(36, 47); // minute
  ctx.stroke();

  // Mechanical Steampunk Dragon Perched on Tower
  // Articulated Copper & Brass Wings
  ctx.fillStyle = '#9e5a32';
  ctx.beginPath();
  ctx.moveTo(32, 24); ctx.lineTo(12, 14); ctx.lineTo(24, 28); ctx.closePath(); // Left wing
  ctx.moveTo(32, 24); ctx.lineTo(52, 14); ctx.lineTo(40, 28); ctx.closePath(); // Right wing
  ctx.fill();

  // Brass Struts & Ribs
  ctx.strokeStyle = '#d4af37';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(32, 24); ctx.lineTo(12, 14);
  ctx.moveTo(22, 22); ctx.lineTo(18, 26);
  ctx.moveTo(32, 24); ctx.lineTo(52, 14);
  ctx.moveTo(42, 22); ctx.lineTo(46, 26);
  ctx.stroke();

  // Dragon Body & Curved Serpentine Neck
  ctx.fillStyle = '#b87333'; // Copper body
  ctx.beginPath();
  ctx.ellipse(32, 26, 8, 5, 0, 0, Math.PI * 2);
  ctx.fill();

  // Dragon Head with Horns
  ctx.fillStyle = '#8a522e';
  ctx.beginPath();
  ctx.moveTo(32, 20); ctx.lineTo(38, 17); ctx.lineTo(34, 23); ctx.closePath();
  ctx.fill();

  // Glowing Molten Amber Furnace Eye
  ctx.fillStyle = '#ffaa00';
  ctx.fillRect(35, 18, 2, 1.5);

  // Gentle Steam Puff from Nostril
  ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.beginPath();
  ctx.arc(42, 16, 2.5, 0, Math.PI * 2);
  ctx.arc(46, 14, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * 6. Universal Prompt Renderer for ANY arbitrary prompt
 */
function renderUniversalPromptScene(ctx, prompt, words, w, h, seed) {
  // Analyze semantic features
  const has = (...terms) => terms.some(t => prompt.includes(t));
  const isNight = has('night', 'dark', 'midnight', 'moon', 'stars');
  const isSunset = has('sunset', 'sunrise', 'golden', 'dusk', 'dawn');
  const isNature = has('forest', 'trees', 'green', 'jungle', 'garden', 'flower', 'meadow');
  const isWater = has('ocean', 'sea', 'beach', 'river', 'water', 'underwater', 'island');
  const isTech = has('car', 'robot', 'mech', 'cyber', 'ship', 'future', 'city', 'building');
  const isAnimal = has('dog', 'puppy', 'wolf', 'lion', 'tiger', 'bear', 'fox', 'bird', 'horse', 'rabbit');
  const isCharacter = has('person', 'woman', 'man', 'girl', 'boy', 'warrior', 'knight', 'wizard');

  // 1. Dynamic Atmosphere & Horizon
  const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.6);
  if (isNight) {
    skyGrad.addColorStop(0, '#060814');
    skyGrad.addColorStop(1, '#1a182c');
  } else if (isSunset) {
    skyGrad.addColorStop(0, '#362145');
    skyGrad.addColorStop(0.5, '#ad4b4e');
    skyGrad.addColorStop(1, '#ffaa55');
  } else if (isNature) {
    skyGrad.addColorStop(0, '#426880');
    skyGrad.addColorStop(1, '#8fbcd4');
  } else if (isWater) {
    skyGrad.addColorStop(0, '#0f3152');
    skyGrad.addColorStop(1, '#2c739e');
  } else {
    skyGrad.addColorStop(0, '#2b374d');
    skyGrad.addColorStop(1, '#6b829e');
  }
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, w, h);

  // 2. Ground / Terrain (Horizon at y=38)
  const groundGrad = ctx.createLinearGradient(0, 38, 0, h);
  if (isWater) {
    groundGrad.addColorStop(0, '#1a4e72');
    groundGrad.addColorStop(1, '#0e2b40');
  } else if (isNature) {
    groundGrad.addColorStop(0, '#2d5232');
    groundGrad.addColorStop(1, '#1b331e');
  } else if (isNight || isTech) {
    groundGrad.addColorStop(0, '#1c1f2b');
    groundGrad.addColorStop(1, '#0d0e14');
  } else {
    groundGrad.addColorStop(0, '#4a443b');
    groundGrad.addColorStop(1, '#26221c');
  }
  ctx.fillStyle = groundGrad;
  ctx.fillRect(0, 38, w, h - 38);

  // 3. Hero Subject Silhouette & Contrast Modeling
  const cx = 32;
  const cy = 34;

  if (isAnimal) {
    // Quadruped / Animal Head & Ears
    ctx.fillStyle = isNight ? '#2a262c' : '#8c593b';
    ctx.beginPath();
    ctx.ellipse(cx, cy, 12, 10, 0, 0, Math.PI * 2);
    ctx.fill();

    // Ears
    ctx.beginPath();
    ctx.moveTo(cx - 9, cy - 6); ctx.lineTo(cx - 10, cy - 15); ctx.lineTo(cx - 4, cy - 9); ctx.closePath();
    ctx.moveTo(cx + 9, cy - 6); ctx.lineTo(cx + 10, cy - 15); ctx.lineTo(cx + 4, cy - 9); ctx.closePath();
    ctx.fill();

    // Expressive Eyes & Nose
    ctx.fillStyle = '#ffcc44';
    ctx.fillRect(cx - 5, cy - 3, 2.5, 2.5);
    ctx.fillRect(cx + 2.5, cy - 3, 2.5, 2.5);
    ctx.fillStyle = '#111111';
    ctx.fillRect(cx - 1, cy + 2, 2, 2);
  } else if (isCharacter) {
    // Humanoid / Portrait
    // Shoulders
    ctx.fillStyle = isNight ? '#1e2433' : '#4a3b4e';
    ctx.beginPath();
    ctx.ellipse(cx, cy + 14, 18, 9, 0, 0, Math.PI * 2);
    ctx.fill();

    // Head
    ctx.fillStyle = '#deb887';
    ctx.beginPath();
    ctx.ellipse(cx, cy - 2, 9, 10, 0, 0, Math.PI * 2);
    ctx.fill();

    // Hair / Crown
    ctx.fillStyle = isNight ? '#00e5ff' : '#4a2e1b';
    ctx.beginPath();
    ctx.arc(cx, cy - 5, 9.5, Math.PI, Math.PI * 2);
    ctx.fill();
  } else if (isTech) {
    // Futuristic Vehicle / Mech Structure
    ctx.fillStyle = '#1d222e';
    ctx.fillRect(cx - 18, cy + 4, 36, 12);
    ctx.fillStyle = '#3a445c';
    ctx.fillRect(cx - 12, cy - 4, 24, 9);
    // Glowing Windshield / Eyes
    ctx.fillStyle = '#00f3ff';
    ctx.fillRect(cx - 10, cy - 2, 20, 4);
  } else {
    // Beautiful Architectural / Focal Monument
    ctx.fillStyle = '#5c4e66';
    ctx.fillRect(cx - 10, cy - 12, 20, 24);
    ctx.fillStyle = '#ffaa44';
    ctx.fillRect(cx - 3, cy - 2, 6, 8);
  }

  // 4. Atmospheric Highlight Bloom & Lighting
  ctx.fillStyle = isNight ? 'rgba(0, 240, 255, 0.35)' : 'rgba(255, 200, 100, 0.35)';
  ctx.beginPath();
  ctx.arc(cx, cy - 4, 16, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Headless CPU fallback procedural pixel buffer generator
 */
function renderHeadlessBuffer(buf, prompt, w, h, seed) {
  const planeSize = w * h;
  for (let y = 0; y < h; y++) {
    const ny = y / h;
    for (let x = 0; x < w; x++) {
      const nx = x / w;
      const idx = (y * w + x) * 4;

      const r = Math.round((Math.sin(nx * 3.14) * 0.5 + 0.5) * 200);
      const g = Math.round((Math.cos(ny * 3.14) * 0.5 + 0.5) * 180);
      const b = Math.round((1.0 - ny) * 220);

      buf[idx] = r;
      buf[idx + 1] = g;
      buf[idx + 2] = b;
      buf[idx + 3] = 255;
    }
  }
}

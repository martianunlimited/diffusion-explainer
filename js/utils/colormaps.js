/**
 * Colormaps for Latent and Cross-Attention Heatmaps
 * Provides Turbo, Magma, Viridis, Inferno, Electric Cyan, and Grayscale.
 */

export const COLORMAPS = {
  TURBO: 'turbo',
  MAGMA: 'magma',
  VIRIDIS: 'viridis',
  INFERNO: 'inferno',
  CYAN: 'cyan',
  GRAY: 'gray'
};

/**
 * Returns [r, g, b] in range [0, 255] for value t in [0, 1]
 */
export function getRGB(t, colormap = COLORMAPS.TURBO) {
  const clampT = Math.max(0, Math.min(1, t));

  switch (colormap) {
    case COLORMAPS.MAGMA:
      return magma(clampT);
    case COLORMAPS.VIRIDIS:
      return viridis(clampT);
    case COLORMAPS.INFERNO:
      return inferno(clampT);
    case COLORMAPS.CYAN:
      return electricCyan(clampT);
    case COLORMAPS.GRAY:
      const v = Math.round(clampT * 255);
      return [v, v, v];
    case COLORMAPS.TURBO:
    default:
      return turbo(clampT);
  }
}

function turbo(x) {
  // Polynomial approximation of Google Turbo colormap
  const r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5668 - x * 58.1375))));
  const g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 - x * 2.7747))));
  const b = 0.1067 + x * (12.5732 - x * (62.4920 - x * (77.8256 - x * (17.3621 - x * 2.3789))));
  return [
    Math.round(Math.max(0, Math.min(1, r)) * 255),
    Math.round(Math.max(0, Math.min(1, g)) * 255),
    Math.round(Math.max(0, Math.min(1, b)) * 255)
  ];
}

function magma(x) {
  const r = Math.sin(x * Math.PI * 0.7);
  const g = Math.pow(x, 2.2);
  const b = Math.sin(x * Math.PI * 0.5) * 0.8 + 0.1;
  return [
    Math.round(Math.max(0, Math.min(1, r * 1.2)) * 255),
    Math.round(Math.max(0, Math.min(1, g * 1.1)) * 255),
    Math.round(Math.max(0, Math.min(1, b)) * 255)
  ];
}

function viridis(x) {
  const r = -0.0019 + x * (0.8354 + x * (-1.3533 + x * 1.5434));
  const g = 0.0049 + x * (1.1062 + x * (1.1895 - x * 1.3486));
  const b = 0.2818 + x * (1.4397 + x * (-2.6845 + x * 1.0505));
  return [
    Math.round(Math.max(0, Math.min(1, r)) * 255),
    Math.round(Math.max(0, Math.min(1, g)) * 255),
    Math.round(Math.max(0, Math.min(1, b)) * 255)
  ];
}

function inferno(x) {
  const r = Math.sin(x * Math.PI * 0.75) * 1.2;
  const g = Math.pow(x, 1.8) * 0.95;
  const b = Math.pow(1 - x, 3) * 0.15 + (x > 0.8 ? (x - 0.8) * 2.5 : 0);
  return [
    Math.round(Math.max(0, Math.min(1, r)) * 255),
    Math.round(Math.max(0, Math.min(1, g)) * 255),
    Math.round(Math.max(0, Math.min(1, b)) * 255)
  ];
}

function electricCyan(x) {
  // Dark deep violet to glowing electric cyan
  const r = Math.round(x * 30 + (x > 0.8 ? (x - 0.8) * 100 : 0));
  const g = Math.round(Math.pow(x, 0.85) * 245);
  const b = Math.round(Math.pow(x, 0.6) * 212 + 40);
  return [Math.min(255, r), Math.min(255, g), Math.min(255, b)];
}

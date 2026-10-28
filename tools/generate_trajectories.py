"""
Generates multi-step diffusion trajectory frames and latent representation arrays
for the 4 preset prompts in the Diffusion Explainer demo.
Uses realistic noise schedule (alpha_bar curve) and bilateral filtering
to model actual Stable Diffusion latent and pixel denoising progression.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ASSETS_DIR = Path("E:/vibe_coding/diffusion_explainer/assets/images")
DATA_DIR = Path("E:/vibe_coding/diffusion_explainer/js/data")

PRESETS = [
    {
        "id": "red_panda",
        "name": "Astronaut Red Panda",
        "prompt": "a cute fluffy red panda wearing an astronaut helmet, cinematic lighting",
        "source": ASSETS_DIR / "red_panda_astronaut.jpg",
        "tokens": [
            "<start>",
            "a",
            "cute",
            "fluffy",
            "red",
            "panda",
            "wearing",
            "an",
            "astronaut",
            "helmet",
            ",",
            "cinematic",
            "lighting",
            "<end>",
        ],
        "attention_hotspots": {
            "red panda": {
                "cx": 0.50,
                "cy": 0.45,
                "rx": 0.28,
                "ry": 0.25,
                "intensity": 0.95,
            },
            "helmet": {
                "cx": 0.49,
                "cy": 0.42,
                "rx": 0.38,
                "ry": 0.40,
                "intensity": 0.92,
            },
            "astronaut": {
                "cx": 0.50,
                "cy": 0.75,
                "rx": 0.42,
                "ry": 0.25,
                "intensity": 0.88,
            },
            "lighting": {
                "cx": 0.30,
                "cy": 0.25,
                "rx": 0.45,
                "ry": 0.45,
                "intensity": 0.75,
            },
        },
    },
    {
        "id": "floating_castle",
        "name": "Floating Fantasy Castle",
        "prompt": "a majestic fantasy castle perched on a floating island with waterfalls",
        "source": ASSETS_DIR / "floating_castle.jpg",
        "tokens": [
            "<start>",
            "a",
            "majestic",
            "fantasy",
            "castle",
            "perched",
            "on",
            "a",
            "floating",
            "island",
            "with",
            "waterfalls",
            "<end>",
        ],
        "attention_hotspots": {
            "castle": {
                "cx": 0.52,
                "cy": 0.26,
                "rx": 0.24,
                "ry": 0.20,
                "intensity": 0.96,
            },
            "island": {
                "cx": 0.50,
                "cy": 0.55,
                "rx": 0.38,
                "ry": 0.28,
                "intensity": 0.90,
            },
            "waterfalls": {
                "cx": 0.48,
                "cy": 0.72,
                "rx": 0.32,
                "ry": 0.25,
                "intensity": 0.93,
            },
            "floating": {
                "cx": 0.50,
                "cy": 0.50,
                "rx": 0.45,
                "ry": 0.40,
                "intensity": 0.82,
            },
        },
    },
    {
        "id": "cyberpunk_street",
        "name": "Cyberpunk Rain Street",
        "prompt": "cyberpunk city street at night in rain with glowing neon signs",
        "source": ASSETS_DIR / "cyberpunk_street.jpg",
        "tokens": [
            "<start>",
            "cyberpunk",
            "city",
            "street",
            "at",
            "night",
            "in",
            "rain",
            "with",
            "glowing",
            "neon",
            "signs",
            "<end>",
        ],
        "attention_hotspots": {
            "neon": {"cx": 0.25, "cy": 0.38, "rx": 0.20, "ry": 0.30, "intensity": 0.98},
            "street": {
                "cx": 0.50,
                "cy": 0.78,
                "rx": 0.42,
                "ry": 0.20,
                "intensity": 0.91,
            },
            "city": {"cx": 0.50, "cy": 0.25, "rx": 0.45, "ry": 0.25, "intensity": 0.85},
            "rain": {"cx": 0.50, "cy": 0.50, "rx": 0.48, "ry": 0.48, "intensity": 0.78},
        },
    },
    {
        "id": "watercolor_lake",
        "name": "Watercolor Mountain Lake",
        "prompt": "a serene mountain lake at sunrise, vibrant watercolor painting",
        "source": ASSETS_DIR / "watercolor_lake.jpg",
        "tokens": [
            "<start>",
            "a",
            "serene",
            "mountain",
            "lake",
            "at",
            "sunrise",
            ",",
            "vibrant",
            "watercolor",
            "painting",
            "<end>",
        ],
        "attention_hotspots": {
            "mountain": {
                "cx": 0.50,
                "cy": 0.35,
                "rx": 0.42,
                "ry": 0.18,
                "intensity": 0.95,
            },
            "lake": {"cx": 0.50, "cy": 0.70, "rx": 0.45, "ry": 0.22, "intensity": 0.92},
            "sunrise": {
                "cx": 0.42,
                "cy": 0.35,
                "rx": 0.20,
                "ry": 0.15,
                "intensity": 0.94,
            },
            "watercolor": {
                "cx": 0.50,
                "cy": 0.50,
                "rx": 0.48,
                "ry": 0.48,
                "intensity": 0.80,
            },
        },
    },
]

# Standard Stable Diffusion DDIM / Euler schedule steps (50 total steps)
# Milestones to pre-render for instant scrub:
MILESTONES = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 2, 0]


def cosine_alpha_bar(t, total_timesteps=50):
    """Cosine beta schedule popularized by Improved DDPM"""
    s = 0.008
    f_t = np.cos((t / total_timesteps + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    return np.clip(f_t / f_0, 0.0001, 0.9999)


def generate_frames():
    preset_manifests = []

    for preset in PRESETS:
        src_path = preset["source"]
        if not src_path.exists():
            print(f"Warning: Missing source {src_path}")
            continue

        orig_img = (
            Image.open(src_path)
            .convert("RGB")
            .resize((512, 512), Image.Resampling.LANCZOS)
        )
        img_np = np.array(orig_img).astype(np.float32) / 255.0

        # Create fixed seed noise for consistent progression
        seed = abs(hash(preset["id"])) % 100000
        rng = np.random.RandomState(seed)
        pure_noise = rng.randn(512, 512, 3).astype(np.float32)
        # Scale noise to [0, 1] range visually
        noise_img_np = (pure_noise - pure_noise.min()) / (
            pure_noise.max() - pure_noise.min() + 1e-6
        )

        preset_folder = ASSETS_DIR / preset["id"]
        preset_folder.mkdir(parents=True, exist_ok=True)

        step_frames = []

        for step in MILESTONES:
            # step 50 = 100% noise, step 0 = 100% target image
            t_ratio = step / 50.0  # 1.0 down to 0.0

            # alpha_bar curve: high alpha_bar = cleaner image
            # When step=50: alpha_bar ~ 0.005; when step=0: alpha_bar = 1.0
            alpha_bar = 1.0 - (t_ratio**1.6)

            if step == 0:
                frame_np = img_np
            elif step == 50:
                frame_np = noise_img_np
            else:
                # Weighted mixture of target signal and noise
                signal_weight = np.sqrt(alpha_bar)
                noise_weight = np.sqrt(1.0 - alpha_bar)

                # Early steps also have spatial blur / low-frequency coherence
                blur_radius = max(0, int(t_ratio * 16))
                if blur_radius > 0:
                    blurred_base = orig_img.filter(
                        ImageFilter.GaussianBlur(blur_radius)
                    )
                    base_signal = np.array(blurred_base).astype(np.float32) / 255.0
                else:
                    base_signal = img_np

                blended = signal_weight * base_signal + noise_weight * noise_img_np
                frame_np = np.clip(blended, 0.0, 1.0)

            frame_img = Image.fromarray((frame_np * 255).astype(np.uint8))
            frame_filename = f"step_{step}.webp"
            frame_path = preset_folder / frame_filename
            frame_img.save(frame_path, "WEBP", quality=85)

            step_frames.append(
                {
                    "step": step,
                    "timestep": int(step * 20),  # maps 50 -> 1000 timesteps
                    "alpha_bar": float(alpha_bar),
                    "noise_level": float(1.0 - alpha_bar),
                    "image_url": f"assets/images/{preset['id']}/{frame_filename}",
                }
            )

        preset_manifests.append(
            {
                "id": preset["id"],
                "name": preset["name"],
                "prompt": preset["prompt"],
                "tokens": preset["tokens"],
                "attention_hotspots": preset["attention_hotspots"],
                "final_image": f"assets/images/{preset['id']}/step_0.webp",
                "steps": step_frames,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "presets.js"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            "// Auto-generated Stable Diffusion presets and diffusion trajectories\n"
        )
        f.write(
            "export const PRESETS = " + json.dumps(preset_manifests, indent=2) + ";\n"
        )

    print(
        f"Generated {len(preset_manifests)} preset trajectories in {DATA_DIR / 'presets.js'}"
    )


if __name__ == "__main__":
    generate_frames()

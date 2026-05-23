# ModelMixSampler (BSS) for ComfyUI

**ModelMixSampler (BSS)** is an advanced, highly optimized custom node for ComfyUI that allows you to chain and alternate denoising steps dynamically between two models of **identical latent architecture** (e.g., SDXL with SDXL, or DiT with DiT, such as HunyuanVideo and Wan2.1) within a single generation process.

By performing direct latent structure and noise phase transitions without slow VAE decoding/encoding cycles, this node delivers maximum sampling speeds, 100% vibrant colors, ultra-crisp details, and completely eliminates any alien color shifts or artifacts.

---

## Key Features

1. **Mix Matching Architectures:**
   Alternate steps between any models with the same latent channels and dimensional structures (e.g., two different 4-channel SDXL models, or two different 16-channel DiT models like Flux, HunyuanVideo, Wan2.1).
2. **Maximum Performance & Zero Artifacts:**
   Since latent dimensions match, step transitions occur directly without slow intermediate pixel-space conversions. Original color fidelity and contrast are perfectly preserved.
3. **Robust Input Validation:**
   The node automatically validates latent channel counts (`c1 == c2`) and spatial dimensions (`dim1 == dim2`) upfront. Mismatching models trigger descriptive errors rather than unexpected `RuntimeError` crashes down the line.
4. **Mathematically Precise Noise Scaling:**
   Invokes single-step denoising wraps, allowing each model's native scheduling mechanism to correctly scale latent noise at each specific step $\sigma_i$.
5. **Compatibility with EPS and Flow/CONST (Flux, Hunyuan, Wan, SDXL):**
   Automatically switches the nature of intermediate noise injection on steps $i > 0$:
   - For **EPS** models (SDXL, SD1.5): passes `zeros` noise, preventing the KSampler from injecting new noise.
   - For **Flow/CONST** models (Flux, HunyuanVideo, Wan2.1): passes the `current_latent` itself as noise, compensating for the built-in $(1 - \sigma)$ multiplier, making the intermediate step a mathematically clean identity transformation.
6. **Unique Step Seeding:**
   Modulates the random seed incrementally at each step (`seed + i`), ensuring that stochastic samplers (like `euler_ancestral` or `dpmpp_sde`) generate fresh noise without phase conflicts or regular artifacts.
7. **GPU VRAM Optimization:**
   Preloads both models to the GPU at once using `comfy.model_management.load_models_gpu([model1, model2])` to avoid slow model swapping/thrashing during sampling.
8. **Flexible Steps Scheduler (Mix Schedule):**
   Supports a robust parsing syntax for custom alternation plans:
   - `1, 2` (alternate steps: 1, 2, 1, 2...)
   - `1, 1, 2, 2` (two steps model 1, two steps model 2...)
   - `1*15, 2*15` (first 15 steps model 1, next 15 steps model 2)
   - `1:10, 2:15` (first 10 steps model 1, next 15 steps model 2)

---

## Installation

1. Copy the folder `ComfyUI-BSS_ModelMixSampler` into your ComfyUI custom nodes directory:
   ```text
   ComfyUI/custom_nodes/
   ```
2. Restart ComfyUI. The nodes will load automatically!

---

## ComfyUI Manager Integration

This package is fully prepared for GitHub publication and submission to the official **ComfyUI Manager** registry:
- Clean module file structure with standard `__init__.py` entry point.
- Precise imports for `ModelMixSamplerNode` and `ModelMixVAEDecodeNode`.
- Unique node mappings (`ModelMixSampler_BSS`, `ModelMixVAEDecode_BSS`).

---

## Node Descriptions

### 1. `ModelMixSampler (BSS)`
The main sampler node for alternating step denoising between matching architectures.

**Parameters:**
- **`model1` / `model2`:** The two models of identical latent structure to mix.
- **`seed`:** Base seed.
- **`steps`:** Total sampling steps.
- **`cfg1` / `cfg2`:** CFG values for each model.
- **`sampler_name1` / `sampler_name2`:** The sampling algorithm for each model.
- **`scheduler`:** Noise level scheduler.
- **`positive1` / `negative1` / `positive2` / `negative2`:** Prompts/conditioning for both models.
- **`latent_image`:** The starting latent image.
- **`denoise`:** Denoising strength (supports img2img).
- **`mix_schedule`:** Step pattern (e.g., `"1, 2"` or `"1*15, 2*15"`).
- **`vae1` / `vae2` (optional):** Retained for full backward compatibility of existing workflows (ignored during the step loop).

### 2. `ModelMixVAEDecode (BSS)`
A smart node designed to decode latents of any dimension (4D or 5D) safely using VAEs of any architecture (2D image or 3D video). Automatically selects the correct VAE based on latent channel count and safely reshapes the tensor dimension, preventing any `IndexError` index crashes.

---

## 📊 Step Schedule Comparison (Mix Schedule)

Below are the comparative generation results using different step alternation patterns:

### Test 1: Mixing EquinoxV4 and ill_realismV4 (SDXL)

You can compare the outputs of the individual models against various step schedules:

| Schedule (Mix Schedule) | Schedule Description | Image |
| --- | --- | --- |
| **EquinoxV4 Only** | Base model EquinoxV4 generation (25 steps) | ![EquinoxV4](images/Test_1/EquinoxV4_25s_cfg6_euler_a_simple.png) |
| **ill_realismV4 Only** | Base model ill_realismV4 generation (25 steps) | ![ill_realismV4](images/Test_1/ill_realismV4_25s_cfg6_euler_a_simple.png) |
| **`1, 2`** | Alternating steps (1, 2, 1, 2...) | ![mode1,2](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode1,2.png) |
| **`2, 1`** | Alternating steps starting with model 2 (2, 1, 2, 1...) | ![mode2,1](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode2,1.png) |
| **`1, 1, 2, 2`** | Two steps of each model consecutively (1, 1, 2, 2...) | ![mode1,1,2,2](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode1,1,2,2.png) |
| **`2, 2, 1, 1`** | Two steps starting with model 2 (2, 2, 1, 1...) | ![mode2,2,1,1](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode2,2,1,1.png) |
| **`1*15, 2*15`** | First 15 steps - Model 1, next 15 steps - Model 2 | ![mode1x15,2x15](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode1x15,%202x15.png) |
| **`2*15, 1*15`** | First 15 steps - Model 2, next 15 steps - Model 1 | ![mode2x15,1x15](images/Test_1/EquinoxV4+ill_realismV4_30s_cfg6_euler_a_simple_mode2x15,%201x15.png) |

---

### Test 2: Mixing ill_realismV4 and WAI16 (SDXL)

You can compare the outputs of the individual models against various step schedules:

| Schedule (Mix Schedule) | Schedule Description | Image |
| --- | --- | --- |
| **ill_realismV4 Only** | Base model ill_realismV4 generation (25 steps) | ![ill_realismV4](images/Test_2/ill_realismV4_25s_cfg6_euler_a_simple.png) |
| **WAI16 Only** | Base model WAI16 generation (25 steps) | ![WAI16](images/Test_2/WAI16_25s_cfg6_euler_a_simple.png) |
| **`1, 2`** | Alternating steps (1, 2, 1, 2...) | ![mode1,2](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode1,2.png) |
| **`2, 1`** | Alternating steps starting with model 2 (2, 1, 2, 1...) | ![mode2,1](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode2,1.png) |
| **`1, 1, 2, 2`** | Two steps of each model consecutively (1, 1, 2, 2...) | ![mode1,1,2,2](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode1,1,2,2.png) |
| **`2, 2, 1, 1`** | Two steps starting with model 2 (2, 2, 1, 1...) | ![mode2,2,1,1](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode2,2,1,1.png) |
| **`1*15, 2*15`** | First 15 steps - Model 1, next 15 steps - Model 2 | ![mode1x15,2x15](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode1x15,2x15.png) |
| **`2*15, 1*15`** | First 15 steps - Model 2, next 15 steps - Model 1 | ![mode2x15,1x15](images/Test_2/ill_realismV4+WAI16_30s_cfg6_euler_a_simple_mode2x15,1x15.png) |

---

## License
© 2026 blacksnowskill (BSS). All rights reserved. Developed by blacksnowskill (BSS).

"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

nodes/node_model_mix_sampler.py
ModelMixSampler — Custom model mixing sampler for ComfyUI.

Alternates generation steps between two models of the same latent structure
(e.g., SDXL and SDXL, HunyuanVideo and HunyuanVideo) with separate CFG and sampler configurations per step.
"""

import logging
from functools import partial
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import comfy.model_sampling
import latent_preview

logger = logging.getLogger("BSS_MODELMIXSAMPLER.node")


def parse_schedule(schedule_str, total_steps):
    """
    Parses the schedule string into a list of model indices (1 or 2) for each step.
    Supports formats like:
      - "1, 2" -> [1, 2, 1, 2...]
      - "1, 1, 2, 2" -> [1, 1, 2, 2, 1, 1, 2, 2...]
      - "1*10, 2*10" -> [1, 1, ..., 2, 2, ...]
      - "1:5, 2:15" -> [1, 1, 1, 1, 1, 2, 2, 2, ...]
    Cycles the resulting pattern to fill the total_steps.
    """
    if not schedule_str or not schedule_str.strip():
        # Default fallback to alternate
        return [1 if i % 2 == 0 else 2 for i in range(total_steps)]

    tokens = [t.strip() for t in schedule_str.split(",")]
    pattern = []
    for token in tokens:
        if not token:
            continue
        if "*" in token:
            parts = token.split("*")
            try:
                val = int(parts[0].strip())
                mult = int(parts[1].strip())
                pattern.extend([val] * mult)
            except ValueError:
                pass
        elif ":" in token:
            parts = token.split(":")
            try:
                val = int(parts[0].strip())
                mult = int(parts[1].strip())
                pattern.extend([val] * mult)
            except ValueError:
                pass
        else:
            try:
                pattern.append(int(token))
            except ValueError:
                pass

    if not pattern:
        # Fallback to alternate if parsing yielded nothing
        return [1 if i % 2 == 0 else 2 for i in range(total_steps)]

    # Make sure we only have valid indices (1 or 2)
    pattern = [1 if x == 1 else 2 for x in pattern]

    # Cycle the pattern to match total_steps
    schedule = []
    for i in range(total_steps):
        schedule.append(pattern[i % len(pattern)])
    return schedule


def is_flow_model(model):
    """
    Checks if the model uses FLOW (constant) architecture.
    """
    ms = model.get_model_object("model_sampling")
    device = comfy.model_management.get_torch_device()
    test_tensor = torch.ones((1, 1, 8, 8), device=device)
    sigma_tensor = torch.tensor([0.5], device=device)
    try:
        res = ms.inverse_noise_scaling(sigma_tensor, test_tensor)
        if not torch.allclose(res, test_tensor, atol=1e-5):
            return True
    except Exception:
        pass
    return False


class SamplerPatchContext:
    def __init__(self):
        self.history = {}
        self.orig_dpmpp_2m = None
        self.orig_dpmpp_2m_sde = None
        
        # References to the current state
        self.current_model = None

    def __enter__(self):
        import comfy.k_diffusion.sampling
        # Save original functions
        self.orig_dpmpp_2m = comfy.k_diffusion.sampling.sample_dpmpp_2m
        self.orig_dpmpp_2m_sde = comfy.k_diffusion.sampling.sample_dpmpp_2m_sde
        
        # Patched DPM-PP 2M
        def patched_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            sigma_fn = lambda t: t.neg().exp()
            t_fn = lambda sigma: sigma.log().neg()
            
            old_denoised_ext = self.history.get("old_denoised_external", None)
            prev_sigma = self.history.get("prev_sigma", None)
            
            latent_format = self.current_model.get_model_object("latent_format")
            
            for i in range(len(sigmas) - 1):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                if callback is not None:
                    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
                    
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = t_next - t
                
                old_denoised = None
                if old_denoised_ext is not None:
                    old_denoised = latent_format.process_in(old_denoised_ext.to(x.device))
                
                if old_denoised is None or prev_sigma is None or sigmas[i + 1] == 0:
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
                else:
                    h_last = t - t_fn(prev_sigma)
                    r = h_last / h
                    denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
                    
                denoised_ext = latent_format.process_out(denoised).clone().cpu()
                self.history["old_denoised_external"] = denoised_ext
                self.history["prev_sigma"] = sigmas[i]
                
            return x

        # Patched DPM-PP 2M SDE
        def patched_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
            if len(sigmas) <= 1:
                return x

            if solver_type not in {'heun', 'midpoint'}:
                raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

            extra_args = {} if extra_args is None else extra_args
            seed = extra_args.get("seed", None)
            
            sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
            
            from comfy.k_diffusion.sampling import BrownianTreeNoiseSampler, sigma_to_half_log_snr, offset_first_sigma_for_snr
            
            noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
            s_in = x.new_ones([x.shape[0]])

            model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
            lambda_fn = partial(sigma_to_half_log_snr, model_sampling=model_sampling)
            sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
            
            old_denoised_ext = self.history.get("old_denoised_external", None)
            h_last = self.history.get("h_last", None)
            
            latent_format = self.current_model.get_model_object("latent_format")

            for i in range(len(sigmas) - 1):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                if callback is not None:
                    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
                
                old_denoised = None
                if old_denoised_ext is not None:
                    old_denoised = latent_format.process_in(old_denoised_ext.to(x.device))

                if sigmas[i + 1] == 0:
                    x = denoised
                    h = None
                else:
                    lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
                    h = lambda_t - lambda_s
                    h_eta = h * (eta + 1)
                    alpha_t = sigmas[i + 1] * lambda_t.exp()

                    x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

                    if old_denoised is not None and h_last is not None:
                        r = h_last / h
                        if solver_type == 'heun':
                            x = x + alpha_t * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                        elif solver_type == 'midpoint':
                            x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

                    if eta > 0 and s_noise > 0:
                        x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

                denoised_ext = latent_format.process_out(denoised).clone().cpu()
                self.history["old_denoised_external"] = denoised_ext
                self.history["h_last"] = h

            return x

        # Replace functions
        comfy.k_diffusion.sampling.sample_dpmpp_2m = patched_dpmpp_2m
        comfy.k_diffusion.sampling.sample_dpmpp_2m_sde = patched_dpmpp_2m_sde
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original functions
        comfy.k_diffusion.sampling.sample_dpmpp_2m = self.orig_dpmpp_2m
        comfy.k_diffusion.sampling.sample_dpmpp_2m_sde = self.orig_dpmpp_2m_sde


class ModelMixSamplerNode:
    """
    ModelMixSampler by BSS.

    Chains denoising steps dynamically between two models of identical latent architecture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000},
                ),
                "cfg1": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "cfg2": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name1": (comfy.samplers.KSampler.SAMPLERS,),
                "sampler_name2": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive1": ("CONDITIONING",),
                "negative1": ("CONDITIONING",),
                "positive2": ("CONDITIONING",),
                "negative2": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mix_schedule": (
                    "STRING",
                    {"default": "1, 2", "multiline": False},
                ),
            },
            "optional": {
                "vae1": ("VAE",),
                "vae2": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample_mix"
    CATEGORY = "BSS/Samplers"
    DESCRIPTION = (
        "ModelMixSampler: Позволяет поочередно использовать две разные модели одной архитектуры на разных шагах генерации. "
        "Поддерживает индивидуальные настройки CFG, разные семплеры и гибкое расписание чередования шагов."
    )

    def sample_mix(
        self,
        model1,
        model2,
        seed,
        steps,
        cfg1,
        cfg2,
        sampler_name1,
        sampler_name2,
        scheduler,
        positive1,
        negative1,
        positive2,
        negative2,
        latent_image,
        denoise,
        mix_schedule,
        vae1=None,
        vae2=None,
    ):
        # 1. Проверяем совместимость архитектур
        latent_format1 = model1.get_model_object("latent_format")
        latent_format2 = model2.get_model_object("latent_format")

        c1 = latent_format1.latent_channels
        c2 = latent_format2.latent_channels
        
        if c1 != c2:
            raise ValueError(
                f"Модели несовместимы по размерности латента! "
                f"Model 1 использует {c1} каналов, а Model 2 использует {c2} каналов. "
                f"Смешивание моделей с разным количеством латентных каналов больше не поддерживается."
            )
            
        dim1 = getattr(latent_format1, "latent_dimensions", 2)
        dim2 = getattr(latent_format2, "latent_dimensions", 2)
        if dim1 != dim2:
            raise ValueError(
                f"Модели несовместимы по пространственной размерности латента! "
                f"Model 1 использует {dim1}D латент, а Model 2 использует {dim2}D латент. "
                f"Смешивание моделей с разным измерением латента не поддерживается."
            )

        # Подготовка латента
        latent_dict = latent_image.copy()
        latent = latent_dict["samples"]
        
        # Фиксируем латентные каналы
        latent = comfy.sample.fix_empty_latent_channels(model1, latent)
        latent_dict["samples"] = latent

        device = comfy.model_management.get_torch_device()

        # 2. Предварительно загружаем обе модели на GPU, чтобы избежать постоянного свопа памяти
        logger.info("[ModelMixSampler] Pre-loading both models to GPU memory...")
        comfy.model_management.load_models_gpu([model1, model2])

        # 3. Расчет sigmas для обеих моделей
        logger.info("[ModelMixSampler] Calculating noise schedule (sigmas) for model 1...")
        dummy_sampler1 = comfy.samplers.KSampler(
            model1,
            steps=steps,
            device=device,
            sampler=sampler_name1,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model1.model_options
        )
        sigmas1 = dummy_sampler1.sigmas.to(device)
        total_steps = len(sigmas1) - 1

        logger.info("[ModelMixSampler] Calculating noise schedule (sigmas) for model 2...")
        dummy_sampler2 = comfy.samplers.KSampler(
            model2,
            steps=steps,
            device=device,
            sampler=sampler_name2,
            scheduler=scheduler,
            denoise=denoise,
            model_options=model2.model_options
        )
        sigmas2 = dummy_sampler2.sigmas.to(device)

        if total_steps <= 0:
            return (latent_dict,)

        # 4. Получаем расписание шагов генерации
        schedule = parse_schedule(mix_schedule, total_steps)
        logger.info(f"[ModelMixSampler] Generated step plan ({total_steps} steps total): {schedule}")

        # 5. Инициализируем элементы интерфейса (прогресс-бар, превью латентов)
        pbar = comfy.utils.ProgressBar(total_steps)
        preview_callback1 = latent_preview.prepare_callback(model1, total_steps)
        preview_callback2 = latent_preview.prepare_callback(model2, total_steps)

        current_latent = latent.clone().to(device)

        # 6. Основной цикл пошаговой генерации
        prev_model_idx = None
        prev_channels = None
        prev_latent_format = None
        latest_denoised = None

        patch_context = SamplerPatchContext()

        with patch_context:
            for i in range(total_steps):
                model_idx = schedule[i]
                if model_idx == 1:
                    current_model = model1
                    current_pos = positive1
                    current_neg = negative1
                    cfg = cfg1
                    sampler_name = sampler_name1
                    preview_callback = preview_callback1
                    curr_channels = c1
                else:
                    current_model = model2
                    current_pos = positive2
                    current_neg = negative2
                    cfg = cfg2
                    sampler_name = sampler_name2
                    preview_callback = preview_callback2
                    curr_channels = c2

                # Обновляем ссылку на текущую модель в контексте
                patch_context.current_model = current_model

                step_seed = seed + i
                latent_format = current_model.get_model_object("latent_format")

                # Подготовка латента (latent_image) и шума (noise) для текущего шага
                if i == 0:
                    # Стартовый шаг: готовим случайный шум, латент равен пустому
                    step_noise = comfy.sample.prepare_noise(current_latent, seed).to(device)
                    step_latent = current_latent
                else:
                    # Извлекаем сигмы для предыдущей и текущей моделей из их собственных шкал
                    sigmas_prev = sigmas1 if prev_model_idx == 1 else sigmas2
                    sigmas_curr_scale = sigmas1 if model_idx == 1 else sigmas2
                    
                    sigma_curr_prev = sigmas_prev[i].item()
                    sigma_curr_to = sigmas_curr_scale[i].item()
                    
                    if latest_denoised is not None:
                        # 1. Вычисляем тип предыдущей модели
                        prev_model = model1 if prev_model_idx == 1 else model2
                        is_from_flow = is_flow_model(prev_model)
                        
                        logger.info(
                            f"[ModelMixSampler] Step {i}: Transition {prev_model_idx} -> {model_idx}. "
                            f"Flow From: {is_from_flow} (sigma_from: {sigma_curr_prev:.4f}), "
                            f"sigma_to: {sigma_curr_to:.4f}"
                        )
                        
                        logger.info(f"[ModelMixSampler] STATS current_latent: mean={current_latent.mean().item():.6f}, std={current_latent.std().item():.6f}, min={current_latent.min().item():.6f}, max={current_latent.max().item():.6f}")
                        logger.info(f"[ModelMixSampler] STATS latest_denoised: mean={latest_denoised.mean().item():.6f}, std={latest_denoised.std().item():.6f}, min={latest_denoised.min().item():.6f}, max={latest_denoised.max().item():.6f}")

                        # 2. Переводим латенты во внутренний масштаб предыдущей модели
                        current_latent_internal = prev_latent_format.process_in(current_latent.to(device))
                        latest_denoised_internal = prev_latent_format.process_in(latest_denoised.to(device))
                        
                        logger.info(f"[ModelMixSampler] STATS current_latent (int): mean={current_latent_internal.mean().item():.6f}, std={current_latent_internal.std().item():.6f}")
                        logger.info(f"[ModelMixSampler] STATS latest_denoised (int): mean={latest_denoised_internal.mean().item():.6f}, std={latest_denoised_internal.std().item():.6f}")

                        # 3. Извлекаем единичный шум ε (≈ N(0,1)) с учетом inverse_noise_scaling
                        if sigma_curr_prev > 1e-5:
                            if is_from_flow:
                                epsilon_old = (current_latent_internal - latest_denoised_internal) * (1.0 - sigma_curr_prev) / sigma_curr_prev
                            else:
                                epsilon_old = (current_latent_internal - latest_denoised_internal) / sigma_curr_prev
                        else:
                            epsilon_old = torch.zeros_like(current_latent_internal)
                            
                        logger.info(f"[ModelMixSampler] STATS epsilon_old: mean={epsilon_old.mean().item():.6f}, std={epsilon_old.std().item():.6f}, min={epsilon_old.min().item():.6f}, max={epsilon_old.max().item():.6f}")

                        # 4. Поскольку архитектуры совпадают, копируем чистую структуру и шум напрямую!
                        x0_new = latest_denoised
                        epsilon_new = epsilon_old
                            
                        logger.info(f"[ModelMixSampler] STATS epsilon_new: mean={epsilon_new.mean().item():.6f}, std={epsilon_new.std().item():.6f}")

                        # 5. Передаём x₀ (ВНЕШНИЙ) как latent и ε как noise.
                        step_latent = x0_new.to(device)
                        step_noise = epsilon_new.to(device)
                    else:
                        logger.warning("[ModelMixSampler] latest_denoised is None! Falling back to raw transition.")
                        step_latent = current_latent
                        step_noise = torch.zeros_like(step_latent).to(device)

                step_denoised = None
                def step_callback(step_idx, denoised, x_tensor, total_steps_inner):
                    nonlocal step_denoised
                    if denoised is not None:
                        step_denoised = denoised.clone()

                # Вызываем сэмплер ComfyUI на 1 шаг
                current_latent = comfy.sample.sample(
                    current_model, step_noise, steps=total_steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    positive=current_pos, negative=current_neg,
                    latent_image=step_latent, denoise=1.0,
                    disable_noise=False, start_step=i, last_step=i + 1,
                    force_full_denoise=False, callback=step_callback,
                    disable_pbar=True, seed=step_seed
                )

                # step_denoised из коллбэка приходит во ВНУТРЕННЕМ масштабе
                if step_denoised is not None:
                    latest_denoised_internal = step_denoised.clone()
                    latest_denoised = latent_format.process_out(latest_denoised_internal)
                else:
                    # Если коллбэк не вернул denoised, используем current_latent (ВНЕШНИЙ масштаб)
                    latest_denoised = current_latent.clone()
                    latest_denoised_internal = latent_format.process_in(latest_denoised)

                logger.info(f"[ModelMixSampler] Step {i} RESULT: current_latent (ext): mean={current_latent.mean().item():.6f}, std={current_latent.std().item():.6f}, min={current_latent.min().item():.6f}, max={current_latent.max().item():.6f}")
                logger.info(f"[ModelMixSampler] Step {i} RESULT: latest_denoised (ext): mean={latest_denoised.mean().item():.6f}, std={latest_denoised.std().item():.6f}, min={latest_denoised.min().item():.6f}, max={latest_denoised.max().item():.6f}")

                # Перемещаем тензоры на промежуточное устройство для оптимизации памяти
                current_latent = current_latent.to(comfy.model_management.intermediate_device())
                latest_denoised = latest_denoised.to(comfy.model_management.intermediate_device())
                latest_denoised_internal = latest_denoised_internal.to(comfy.model_management.intermediate_device())

                prev_model_idx = model_idx
                prev_channels = curr_channels
                prev_latent_format = latent_format
                pbar.update_absolute(i + 1, total_steps)
                
                # Передаем в превью-коллбэк во ВНУТРЕННЕМ масштабе модели
                preview_callback(i, latest_denoised_internal, latest_denoised_internal, total_steps)

        # 7. Возвращаем результат
        out_latent = latent_dict.copy()
        out_latent["samples"] = current_latent.cpu()

        logger.info("[ModelMixSampler] Generation finished.")
        return (out_latent,)


class ModelMixVAEDecodeNode:
    """
    ModelMixVAEDecode by BSS.

    Decodes latent images back into pixel space. Automatically selects the correct VAE
    and intelligently adapts dimensions (4D <-> 5D) to prevent IndexError on 3D/video VAEs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The mixed latent to be decoded."}),
                "vae1": ("VAE", {"tooltip": "Primary VAE."}),
            },
            "optional": {
                "vae2": ("VAE", {"tooltip": "Secondary VAE (optional)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "BSS/Samplers"
    DESCRIPTION = (
        "ModelMixVAEDecode: Декодирует латент в изображение. Автоматически выбирает VAE "
        "(vae1 или vae2) в зависимости от каналов входящего латента и безопасно адаптирует размерность."
    )

    def decode(self, samples, vae1, vae2=None):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]

        channels = latent.shape[1]
        device = comfy.model_management.get_torch_device()

        # Выбираем подходящий VAE на основе каналов латента
        selected_vae = vae1
        if vae2 is not None:
            c1 = getattr(vae1, "latent_channels", 4)
            c2 = getattr(vae2, "latent_channels", 4)
            if channels == c2 and channels != c1:
                selected_vae = vae2
                logger.info(f"[ModelMixVAEDecode] Selected vae2 ({c2} channels) because latent has {channels} channels.")
            else:
                logger.info(f"[ModelMixVAEDecode] Selected vae1 ({c1} channels) because latent has {channels} channels.")
        else:
            logger.info("[ModelMixVAEDecode] Only vae1 is provided. Using vae1.")

        # Копируем латент на устройство
        latent_processed = latent.clone().to(device)
        vae_latent_dim = getattr(selected_vae, "latent_dim", 2)

        # Адаптация 4D <-> 5D на случай, если VAE 3D (например WanVAE), а латент 4D
        if vae_latent_dim == 3 and latent_processed.ndim == 4:
            logger.info("[ModelMixVAEDecode] Selected VAE is 3D, but latent is 4D. Unsqueezing time dimension to 5D [B, C, 1, H, W].")
            latent_processed = latent_processed.unsqueeze(2)
        elif vae_latent_dim == 2 and latent_processed.ndim == 5:
            logger.info("[ModelMixVAEDecode] Selected VAE is 2D, but latent is 5D. Squeezing time dimension to 4D [B, C, H, W].")
            latent_processed = latent_processed[:, :, 0]

        # Декодирование
        images = selected_vae.decode(latent_processed)

        # Объединение кадров/батчей для 5D-выхода видео VAE
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

        return (images.to(comfy.model_management.intermediate_device()),)

"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

ComfyUI-BSS_ModelMixSampler — Model Mix Sampler node pack.
Provides step-by-step alternating sampling using two different models.
"""

import logging

logger = logging.getLogger("BSS_MODELMIXSAMPLER")

try:
    from .nodes.node_model_mix_sampler import ModelMixSamplerNode, ModelMixVAEDecodeNode
    _load_error = None
except Exception as e:
    _load_error = e
    logger.error(f"[BSS_MODELMIXSAMPLER] Failed to load nodes: {e}", exc_info=True)
    ModelMixSamplerNode = None
    ModelMixVAEDecodeNode = None

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if ModelMixSamplerNode is not None:
    NODE_CLASS_MAPPINGS["ModelMixSampler_BSS"] = ModelMixSamplerNode
    NODE_DISPLAY_NAME_MAPPINGS["ModelMixSampler_BSS"] = "ModelMixSampler (BSS)"

if ModelMixVAEDecodeNode is not None:
    NODE_CLASS_MAPPINGS["ModelMixVAEDecode_BSS"] = ModelMixVAEDecodeNode
    NODE_DISPLAY_NAME_MAPPINGS["ModelMixVAEDecode_BSS"] = "ModelMixVAEDecode (BSS)"

if _load_error:
    logger.warning(f"[BSS_MODELMIXSAMPLER] Partial load due to error: {_load_error}")
else:
    loaded = list(NODE_CLASS_MAPPINGS.keys())
    logger.info(f"[BSS_MODELMIXSAMPLER] Loaded {len(loaded)} nodes: {loaded} | Version: v1.0.0 | Authorship: blacksnowskill (BSS)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

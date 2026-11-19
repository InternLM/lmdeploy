"""Opt-in vision PatchEmbed Conv3d contract; no normalization changes.

Default off. Set before model import in fresh candidate workers only.
The enabled forward delegates to the unchanged public Glm4v parent method.
"""
import os

ENV_NAME = "LMDEPLOY_GLM53_VISION_XT_PATCH_EMBED"
ENABLED = os.environ.get(ENV_NAME, "0").strip().lower() in {"1", "true", "yes", "on"}

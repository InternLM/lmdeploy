# Copyright (c) OpenMMLab. All rights reserved.
"""FlashAttention-3 build capability detection."""

import torch


def _get_build_flags() -> dict | None:
    """Read build flags exported by recent FlashAttention-3 wheels."""
    try:
        from flash_attn_config import CONFIG
        flags = CONFIG['build_flags']
    except (ImportError, KeyError, TypeError):
        return None
    return flags if isinstance(flags, dict) else None


def fa3_build_supports_head_dims(head_size: int, v_head_size: int | None = None) -> bool:
    """Return whether the installed FA3 wheel contains a head shape.

    FA3 wheels may omit template instantiations to reduce build time and binary size. Wheels without exported build
    metadata are accepted only for symmetric head shapes, preserving compatibility with older packages while avoiding
    unsafe asymmetric dispatch.
    """
    if v_head_size is None:
        v_head_size = head_size

    flags = _get_build_flags()
    if flags is None:
        return head_size == v_head_size and head_size <= 256

    def _disabled(name: str) -> bool:
        # Older generated configs used FLASH_ATTENTION for HDIMDIFF while the
        # other flags use FLASHATTENTION. Accept both spellings.
        aliases = (name, name.replace('FLASHATTENTION_DISABLE_HDIMDIFF',
                                      'FLASH_ATTENTION_DISABLE_HDIMDIFF'))
        return any(bool(flags.get(alias, False)) for alias in aliases)

    if head_size <= 64:
        if _disabled('FLASHATTENTION_DISABLE_HDIM64'):
            return False
        return v_head_size <= 64 or (
            v_head_size <= 512 and not _disabled('FLASHATTENTION_DISABLE_HDIMDIFF64'))
    if head_size <= 96:
        return v_head_size <= 96 and not _disabled('FLASHATTENTION_DISABLE_HDIM96')
    if head_size <= 128:
        return v_head_size <= 128 and not _disabled('FLASHATTENTION_DISABLE_HDIM128')
    if head_size <= 192:
        if _disabled('FLASHATTENTION_DISABLE_HDIM192'):
            return False
        if v_head_size <= 128:
            return not _disabled('FLASHATTENTION_DISABLE_HDIMDIFF192')
        return v_head_size <= 192
    if head_size <= 256:
        return v_head_size <= 256 and not _disabled('FLASHATTENTION_DISABLE_HDIM256')
    return False


def fa3_build_supports_operation(
    head_size: int,
    v_head_size: int | None = None,
    *,
    device_capability: tuple[int, int],
    dtype: torch.dtype | None = None,
    paged_kv: bool = False,
    varlen: bool = False,
    local: bool = False,
    softcap: bool = False,
) -> bool:
    """Return whether the installed FA3 build can run an operation.

    Shape templates alone are insufficient: FA3 wheels can independently
    omit paged-KV, varlen, local-attention, softcap, FP16, and SM8x kernels.
    FA3 also rejects asymmetric QK/V head dimensions on non-Hopper devices
    before template dispatch.
    """
    if v_head_size is None:
        v_head_size = head_size

    major, _ = device_capability
    if major < 8:
        return False
    if head_size != v_head_size and major != 9:
        return False
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        return False
    if not fa3_build_supports_head_dims(head_size, v_head_size):
        return False

    flags = _get_build_flags()
    if flags is None:
        # Older wheels do not expose feature-level build metadata. Preserve
        # their established symmetric-shape behavior; asymmetric shapes were
        # already rejected by fa3_build_supports_head_dims above.
        return True

    def _disabled(name: str) -> bool:
        return bool(flags.get(name, False))

    if major == 8 and _disabled('FLASHATTENTION_DISABLE_SM8x'):
        return False
    if dtype == torch.float16 and _disabled('FLASHATTENTION_DISABLE_FP16'):
        return False
    if paged_kv and _disabled('FLASHATTENTION_DISABLE_PAGEDKV'):
        return False
    if varlen and _disabled('FLASHATTENTION_DISABLE_VARLEN'):
        return False
    if local and _disabled('FLASHATTENTION_DISABLE_LOCAL'):
        return False
    if softcap and _disabled('FLASHATTENTION_DISABLE_SOFTCAP'):
        return False
    return True

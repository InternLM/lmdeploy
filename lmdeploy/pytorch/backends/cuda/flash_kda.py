"""Optional native FlashKDA adapter with explicit intermediate-state outputs."""

import torch


def check_flash_kda():
    """Fail before inference if the installed native operator lacks IC outputs."""
    import flash_kda

    arguments = {argument.name for argument in torch.ops.flash_kda.fwd.default._schema.arguments}
    if not {'checkpoint_state', 'checkpoint_offsets'} <= arguments:
        raise RuntimeError('FlashKDA must provide native checkpoint_state and checkpoint_offsets outputs.')
    return flash_kda


def flash_kda_prefill(query, key, value, gate, beta, a_log, dt_bias,
                      initial_state, cu_seqlens, lower_bound, checkpoint=None):
    """Use the same native prefill path with or without checkpoint export."""
    import flash_kda

    if query.dtype != torch.bfloat16 or query.size(-1) != 128 or initial_state.dtype != torch.float32:
        raise ValueError('Native FlashKDA prefill requires BF16 inputs, D=128 and FP32 recurrent state.')
    if lower_bound is None:
        raise ValueError('Native FlashKDA requires a finite gate lower bound.')
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    final_state = torch.empty_like(initial_state, memory_format=torch.contiguous_format)
    workspace = torch.empty(
        flash_kda.get_workspace_size(query.size(1), query.size(2), initial_state.size(0)),
        dtype=torch.uint8, device=query.device)
    torch.ops.flash_kda.fwd(
        query.contiguous(), key.contiguous(), value.contiguous(), gate.contiguous(), beta.contiguous(),
        query.size(-1) ** -0.5, output, workspace, a_log,
        dt_bias.reshape(query.size(2), query.size(3)), lower_bound,
        initial_state=initial_state.contiguous(), final_state=final_state, cu_seqlens=cu_seqlens,
        checkpoint_state=None if checkpoint is None else checkpoint.recurrent_state_out,
        checkpoint_offsets=None if checkpoint is None else checkpoint.offsets,
    )
    return output, final_state

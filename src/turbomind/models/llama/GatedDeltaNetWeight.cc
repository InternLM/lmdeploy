#include "src/turbomind/models/llama/GatedDeltaNetWeight.h"

namespace turbomind {

GatedDeltaNetWeight::GatedDeltaNetWeight(int      hidden_dim,
                                         int      num_k_heads,
                                         int      num_v_heads,
                                         int      key_head_dim,
                                         int      value_head_dim,
                                         int      d_conv,
                                         bool     bias,
                                         int      tp_size,
                                         int      tp_rank,
                                         DataType data_type,
                                         DataType weight_type,
                                         int      group_size):
    tp_rank_(tp_rank), tp_size_(tp_size)
{
    // Dimensions per HuggingFace Qwen3_5GatedDeltaNet:
    //   key_dim   = num_k_heads * key_head_dim
    //   value_dim = num_v_heads * value_head_dim
    // For TP>1, divide output dimensions by tp_size to match Python save_split.
    const int key_dim    = num_k_heads * key_head_dim / tp_size;
    const int value_dim  = num_v_heads * value_head_dim / tp_size;
    const int v_heads_tp = num_v_heads / tp_size;

    // GatedDeltaNet projections are stored as DENSE (non-quantized) weights in
    // the checkpoint, even for AWQ models. Use data_type (not weight_type) so
    // LlamaDenseWeight registers the parameter as "weight" (not "qweight") and
    // the model loading can match the saved file names correctly.
    const DataType dense_wtype = data_type;
    const int      dense_gsz   = 0;

    // in_proj_qkv: hidden_dim -> key_dim * 2 + value_dim  (Q and K use key_dim, V uses value_dim)
    in_proj_qkv.emplace(hidden_dim, key_dim * 2 + value_dim, data_type, bias, dense_wtype, dense_gsz);

    // in_proj_z: hidden_dim -> value_dim  (output gating signal, reshaped to num_v_heads * value_head_dim)
    in_proj_z.emplace(hidden_dim, value_dim, data_type, bias, dense_wtype, dense_gsz);

    // in_proj_b: hidden_dim -> num_v_heads/tp  (per-head beta scalar, passed through sigmoid)
    in_proj_b.emplace(hidden_dim, v_heads_tp, data_type, bias, dense_wtype, dense_gsz);

    // in_proj_a: hidden_dim -> num_v_heads/tp  (per-head alpha/dt scalar, combined with A_log and dt_bias)
    in_proj_a.emplace(hidden_dim, v_heads_tp, data_type, bias, dense_wtype, dense_gsz);

    // out_proj: value_dim -> hidden_dim
    out_proj.emplace(value_dim, hidden_dim, data_type, bias, dense_wtype, dense_gsz);

    // Register dense weight sub-modules with tp_rank for name suffix alignment
    register_module("in_proj_qkv", in_proj_qkv, tp_rank_);
    register_module("in_proj_z", in_proj_z, tp_rank_);
    register_module("in_proj_b", in_proj_b, tp_rank_);
    register_module("in_proj_a", in_proj_a, tp_rank_);
    register_module("out_proj", out_proj, tp_rank_);

    // conv1d: depthwise conv weights, shape (conv_dim, d_conv)
    const int conv_dim = key_dim * 2 + value_dim;
    conv1d             = Tensor{{conv_dim, d_conv}, data_type, kDEVICE};
    register_parameter("conv1d." + std::to_string(tp_rank_) + ".weight", conv1d);

    // A_log: log-space decay per head, shape (num_v_heads/tp,)
    A_log = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("A_log." + std::to_string(tp_rank_) + ".weight", A_log);

    // dt_bias: per head, shape (num_v_heads/tp,)
    dt_bias = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("dt_bias." + std::to_string(tp_rank_) + ".weight", dt_bias);

    // norm: RMSNormGated weight, shape (value_head_dim,)
    norm = Tensor{{value_head_dim}, data_type, kDEVICE};
    register_parameter("norm.weight", norm);
}

void GatedDeltaNetWeight::prepare()
{
    in_proj_qkv.prepare();
    in_proj_z.prepare();
    in_proj_b.prepare();
    in_proj_a.prepare();
    out_proj.prepare();
}

}  // namespace turbomind

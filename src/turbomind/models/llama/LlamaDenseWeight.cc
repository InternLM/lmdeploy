#include "src/turbomind/models/llama/LlamaDenseWeight.h"

namespace turbomind {

void LlamaDenseWeight::emplace(
    int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size)
{
    this->data_type   = data_type;
    this->weight_type = weight_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    this->group_size  = group_size;

    const auto wbits = core::get_byte_size(weight_type, 8);

    weight = core::Tensor({input_dim, output_dim}, weight_type, MEMORY_GPU);
    register_parameter(wbits < 16 ? "qweight" : "weight", weight);

    if (bias) {
        this->bias = core::Tensor{{output_dim}, data_type, MEMORY_GPU};
        register_parameter("bias", this->bias);
    }

    if (wbits < 16) {
        TM_CHECK(input_dim % group_size == 0) << input_dim << " " << group_size;
        scales = core::Tensor{{input_dim / group_size, output_dim}, data_type, MEMORY_GPU};
        zeros  = core::Tensor{{input_dim / group_size, output_dim}, data_type, MEMORY_GPU};
        register_parameter("scales", scales);
        register_parameter("zeros", zeros);
    }
}

LlamaAttentionWeight::LlamaAttentionWeight(int      hidden_dim,
                                           int      head_dim,
                                           int      head_num,
                                           int      kv_head_num,
                                           MLAParam mla,
                                           bool     bias,
                                           bool     qk_norm,
                                           int      tp_size,
                                           int      tp_rank,
                                           DataType data_type,
                                           DataType weight_type,
                                           int      group_size)
{
    if (mla.kv_lora_rank == 0) {
        qkv.emplace(
            hidden_dim, (head_num + 2 * kv_head_num) * head_dim / tp_size, data_type, bias, weight_type, group_size);
        register_module("w_qkv", qkv, tp_rank);
        if (qk_norm) {
            q_a_layernorm  = core::Tensor{{head_dim}, data_type, MEMORY_GPU};
            kv_a_layernorm = core::Tensor{{head_dim}, data_type, MEMORY_GPU};
            register_parameter("q_norm", q_a_layernorm);
            register_parameter("k_norm", kv_a_layernorm);
        }
    }
    else {
        const int qk_nope_dim = head_dim - mla.qk_rope_dim;
        if (mla.q_lora_rank) {
            q_a_proj.emplace(hidden_dim, mla.q_lora_rank, data_type, false, weight_type, group_size);
            q_b_proj.emplace(mla.q_lora_rank, head_num * head_dim / tp_size, data_type, false, weight_type, group_size);
            q_a_layernorm = core::Tensor{{q_b_proj.input_dim}, data_type, MEMORY_GPU};
            register_module("q_a_proj", q_a_proj);
            register_module("q_b_proj", q_b_proj, tp_rank);
            register_parameter("q_a_layernorm", q_a_layernorm);
        }
        else {
            q_proj.emplace(hidden_dim, head_num * head_dim / tp_size, data_type, false, weight_type, group_size);
            register_module("q_proj", q_proj, tp_rank);
        }
        kv_a_proj.emplace(hidden_dim, mla.kv_lora_rank + mla.qk_rope_dim, data_type, false, weight_type, group_size);
        kv_b_proj.emplace(mla.kv_lora_rank,
                          head_num * (qk_nope_dim + mla.v_head_dim) / tp_size,
                          data_type,
                          false,
                          weight_type,
                          group_size);

        kv_a_layernorm = core::Tensor{{kv_b_proj.input_dim}, data_type, MEMORY_GPU};
        register_module("kv_a_proj", kv_a_proj);
        register_module("kv_b_proj", kv_b_proj, tp_rank);
        register_parameter("kv_a_layernorm", kv_a_layernorm);
    }
    output.emplace((head_num * head_dim) / tp_size, hidden_dim, data_type, bias, weight_type, group_size);
    register_module("wo", output, tp_rank);
}

LlamaFfnWeight::LlamaFfnWeight(int      hidden_dim,
                               int      inter_size,
                               int      tp_size,
                               int      tp_rank,
                               DataType data_type,
                               DataType weight_type,
                               int      group_size,
                               bool     fuse_silu_act)
{
    TM_CHECK(inter_size % tp_size == 0) << inter_size << " " << tp_size;

    inter_size /= tp_size;

    this->inter_size = inter_size;

    gating.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size);

    intermediate.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size);

    // fused_gating_intermediate = {hidden_dim, inter_size * 2, data_type, weight_type, group_size};
    is_fused_silu = fuse_silu_act;

    output.emplace(inter_size, hidden_dim, data_type, false, weight_type, group_size);

    register_module("w1", gating, tp_rank);
    register_module("w3", intermediate, tp_rank);
    register_module("w2", output, tp_rank);
}

MoeFfnWeight::MoeFfnWeight(int             layer_id,
                           const MoeParam& param,
                           int             hidden_dim,
                           DataType        data_type,
                           DataType        weight_type,
                           int             group_size,
                           int             tp_size,
                           int             tp_rank,
                           bool            fuse_silu_act)
{
    if ((int)param.expert_num.size() <= layer_id) {
        return;
    }

    const int expert_num = param.expert_num[layer_id];

    if (expert_num == 0) {
        return;
    }

    // printf("%d %d %d\n", (int)hidden_dim, (int)param.inter_size, (int)expert_num);

    gate.emplace(hidden_dim, expert_num, data_type, false, data_type, 1);
    register_module("gate", gate);

    method        = param.method;
    fuse_silu_act = fuse_silu_act && method == MoeParam::kFused;

    experts.reserve(expert_num);
    for (int i = 0; i < expert_num; ++i) {
        experts.emplace_back(new LlamaFfnWeight{
            hidden_dim, param.inter_size, tp_size, tp_rank, data_type, weight_type, group_size, fuse_silu_act});
        register_module("experts", *experts.back(), i);
    }

    if (param.shared_gate) {
        shared_gate.emplace(hidden_dim, 1, data_type, false, data_type, 1);
        register_module("shared_gate", shared_gate);
    }
}

}  // namespace turbomind
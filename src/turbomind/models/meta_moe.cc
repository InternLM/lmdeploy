// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/meta_moe.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/moe_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include <unordered_map>
#include <vector>

namespace turbomind {

static core::LinearConfig linear_cfg_from(const LinearWeight& src)
{
    core::LinearConfig c;
    c.input_dim  = src.input_dim;
    c.output_dim = src.output_dim;
    c.data_type  = src.data_type;
    c.format     = src.weight_format;
    c.has_bias   = static_cast<bool>(src.bias);
    return c;
}

static void alias_linear(LinearWeight& dst, const LinearWeight& src)
{
    dst.weight = src.weight;
    dst.bias   = src.bias;
    dst.scales = src.scales;
    dst.zeros  = src.zeros;
    src.copy_metadata_to(dst);
}

void alias_routed_moe(MoeWeight& dst, const MoeWeight& donor)
{
    TM_CHECK(donor.gate && donor.experts);
    TM_CHECK(!dst.gate && !dst.experts);
    TM_CHECK_NOTNULL(donor.expert(0));

    dst.create_child("gate", linear_cfg_from(*donor.gate));
    alias_linear(*dst.gate, *donor.gate);

    dst.create_child("experts", core::ModuleListConfig{});
    for (int e = 0; e < donor.expert_num; ++e) {
        auto*           src = TM_CHECK_NOTNULL(donor.expert(e));
        core::FfnConfig fc;
        fc.hidden_dim     = src->hidden_dim;
        fc.inter_size     = src->inter_size * src->tp_size;  // ctor divides by tp
        fc.tp_size        = src->tp_size;
        fc.tp_rank        = src->tp_rank;
        fc.is_expert      = true;
        fc.data_type      = src->w2 ? src->w2->data_type : (src->w1w3 ? src->w1w3->data_type : DataType{});
        fc.fuse_silu      = src->is_fused_silu;
        fc.act_type       = static_cast<int>(src->act_type);
        auto* ffn         = static_cast<FfnWeight*>(dst.experts->create_child(std::to_string(e), fc));
        auto  alias_child = [&](const char* name, LinearWeight* s) {
            if (!s) {
                return;
            }
            ffn->create_child(name, linear_cfg_from(*s));
            alias_linear(*static_cast<LinearWeight*>(ffn->child(name)), *s);
        };
        if (src->w1w3) {
            alias_child("w1w3", src->w1w3.get());
        }
        else {
            alias_child("w1", src->w1.get());
            alias_child("w3", src->w3.get());
        }
        alias_child("w2", src->w2.get());
    }
    // never touch shared_gate
}

bool ModelHasMetaMoe(const ModelWeight& model)
{
    for (auto* dl : model.layers_list()) {
        if (dl->moe_ffn && dl->moe_ffn->meta_group >= 0) {
            return true;
        }
    }
    return false;
}

void PrepareMetaMoe(ModelWeight& model)
{
    const int n = static_cast<int>(model.layers_list().size());
    for (int i = 0; i < n; ++i) {
        auto* dl = model.layer(i);
        if (dl->attention) {
            dl->attention->prepare();
        }
        if (dl->linear_attn) {
            dl->linear_attn->prepare();
        }
        if (dl->attention_norm) {
            dl->attention_norm->prepare();
        }
        if (dl->ffn_norm) {
            dl->ffn_norm->prepare();
        }
        if (dl->feed_forward) {
            dl->feed_forward->prepare();
        }
        if (dl->moe_ffn) {
            dl->moe_ffn->prepare_routed_linears();
        }
    }
    if (model.norm) {
        model.norm->prepare();
    }
    if (model.output) {
        model.output->prepare();
    }

    std::unordered_map<int, MoeWeight*>              donors;
    std::unordered_map<int, std::vector<MoeWeight*>> groups;
    for (int i = 0; i < n; ++i) {
        auto* moe = model.layer(i)->moe_ffn.get();
        if (!moe || moe->meta_group < 0) {
            continue;
        }
        groups[moe->meta_group].push_back(moe);
        if (moe->is_meta_donor) {
            TM_CHECK_EQ(donors.count(moe->meta_group), 0);
            donors[moe->meta_group] = moe;
        }
    }
    for (auto& [g, members] : groups) {
        auto* donor = TM_CHECK_NOTNULL(donors.at(g));
        for (auto* dst : members) {
            if (dst != donor) {
                alias_routed_moe(*dst, *donor);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (auto* moe = model.layer(i)->moe_ffn.get()) {
            moe->link_block();
        }
    }
}

}  // namespace turbomind

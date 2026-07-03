
#include "src/turbomind/models/output_processor.h"

#include <functional>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/cross_entropy_kernels.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::shared_ptr;

struct OutputProcessor::Impl {

    static constexpr auto kAll = GenerationConfig::kAll;

    const int vocab_size_;
    const int max_logits_len_;
    const int tp_rank_;

    std::function<Tensor(const Tensor&)> lm_head_;

    Impl(int vocab_size, int max_logits_len, int tp_rank, int phases, std::function<Tensor(const Tensor&)> lm_head):
        vocab_size_{vocab_size}, max_logits_len_{max_logits_len}, tp_rank_{tp_rank}, lm_head_{std::move(lm_head)}
    {
        for (int i = 0; i < phases; ++i) {
            data_.emplace_back();
        }
    }

    struct Data {
        Interval full_states;  // requested range for full hidden states
        Interval full_logits;  // requested range for full logits

        vector<std::tuple<int, int, Interval, Interval>> output_states;
        vector<std::tuple<int, int, Interval, Interval>> output_logits;

        Interval full_ce_loss;  // requested range for CE loss logits
        // Per CE scoring segment: (request_idx, range). `range` is the hidden-buffer position
        // interval; `ce_targets` is indexed by that same position, so no extra offset is stored.
        vector<std::tuple<int, Interval>> ce_loss_segments;
        Buffer_<int>                      ce_targets;
    };

    vector<Data> data_;

    struct Matching {
        Interval& target;
        const int offset_d;
        Interval  src;
        Interval  dst;

        bool operator()(const Interval& x, int offset_s, Interval& merged)
        {
            if (auto y = target & x; y && y.begin() == target.begin()) {
                dst    = {y.begin() - offset_d, y.size()};
                src    = {offset_s + (y.begin() - x.begin()), y.size()};
                merged = merged | src;
                target = -(int)y.size() | target;
                return true;
            }
            return false;
        }
    };

    void Add(int phase, TensorMap& env)
    {
        const Buffer_<RequestCache*> rc = env.at("requests").buffer();

        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& r = *c.req;
            auto& g = r.gen_cfg;
            if (g.output_logits) {
                c.output_logits = g.output_logits == kAll ? Interval{c.step0} : Interval{c.prompt_len - 1};
                c.logits_offset = c.output_logits.begin();
            }
            if (g.return_ppl) {
                c.input_ce_loss = {c.step0, c.prompt_len - 1};
            }
            if (g.output_last_hidden_state) {
                c.output_hidden_states =
                    g.output_last_hidden_state == kAll ? Interval{c.step0} : Interval{c.prompt_len - 1};
                c.hidden_states_offset = c.output_hidden_states.begin();
                // dbg(&c.output_hidden_states, c.hidden_states_offset);
            }
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);

        const auto& rc = env.at("batch").data<BatchData*>()[0]->rc;

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        vector<Interval> all_tokens;
        vector<Interval> sel_tokens;
        bool             has_ce = false;
        for (int i = 0; i < rc.size(); ++i) {
            using Size = Interval::Size;
            auto& c    = *rc[i];
            all_tokens.emplace_back(c.history_len + c.alpha, Size{c.input_len});
            sel_tokens.emplace_back(c.history_len + c.alpha + c.input_len - 1, Size{1});
            if (!c.generating) {
                sel_tokens.back() = {};
            }
            has_ce = has_ce || (bool)c.input_ce_loss;
            // dbg(&all_tokens.back(), &sel_tokens.back());
        }

        const int token_num = *env.at("token_num").data<int>();

        if (has_ce && tp_rank_ == 0) {
            // Lazily grow the per-phase device target buffer to this step's token count (<= token_num).
            if (d.ce_targets.size() < token_num) {
                d.ce_targets = {token_num, kDEVICE};
            }
        }

        d.full_logits = {INT_MAX, 0};
        d.full_states = {INT_MAX, 0};

        Interval select_states{INT_MAX, 0};
        Interval select_logits{INT_MAX, 0};

        d.output_logits = {};
        d.output_states = {};

        d.full_ce_loss     = {INT_MAX, 0};
        d.ce_loss_segments = {};

        int offset = 0;

        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            auto& g = c.req->gen_cfg;
            if (c.output_hidden_states) {
                Matching m{c.output_hidden_states, c.hidden_states_offset};
                int      type = 0;
                if (m(sel_tokens[i], i, select_states)) {
                    type = 1;
                }
                else if (m(all_tokens[i], offset, d.full_states)) {
                    type = 2;
                }
                if (type) {
                    d.output_states.emplace_back(i, type, m.src, m.dst);
                    // dbg(type, &m.src, &m.dst);
                }
            }
            if (c.output_logits) {
                Matching m{c.output_logits, c.logits_offset};
                int      type = 0;
                if (m(sel_tokens[i], i, select_logits)) {
                    type = 1;
                }
                else if (m(all_tokens[i], offset, d.full_logits)) {
                    type = 2;
                }
                if (type) {
                    d.output_logits.emplace_back(i, type, m.src, m.dst);
                }
            }
            if (c.input_ce_loss) {
                if (tp_rank_ == 0 && !c.ce_loss) {
                    // Per-request accumulator, allocated and zeroed once.
                    c.ce_loss = {1, kDEVICE};
                    Clear(c.ce_loss);
                }
                Matching m{c.input_ce_loss, /* offset_d */ 0};
                if (m(all_tokens[i], offset, d.full_ce_loss)) {
                    if (tp_rank_ == 0) {
                        copy(c.token_ids + m.dst.begin() + 1, (int)m.src.size(), d.ce_targets.data() + m.src.begin());
                    }
                    d.ce_loss_segments.emplace_back(i, m.src);
                }
            }
            offset += c.input_len;
        }

        // logits depends on hidden states
        d.full_states = d.full_states | d.full_logits | d.full_ce_loss;
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d = data_.at(phase);
        if (d.full_states) {
            env.produce("output_hidden_states", Tensor{});
        }
    }

    void ComputeCeLoss(Data& data, const Tensor& logits, int base, const vector<shared_ptr<RequestCache>>& rs)
    {
        if (tp_rank_ != 0 || data.ce_loss_segments.empty()) {
            return;
        }

        const auto     stream = core::Context::stream().handle();
        const Interval rows{base, Interval::Size{(int)logits.shape(0)}};

        for (const auto& [request_idx, range] : data.ce_loss_segments) {
            if (auto src = range & rows) {
                const int tokens        = (int)src.size();
                const int target_offset = src.begin();
                const int logit_offset  = src.begin() - base;
                invokeCrossEntropyLoss(rs[request_idx]->ce_loss.data(),
                                       logits,
                                       data.ce_targets.data(),
                                       target_offset,
                                       logit_offset,
                                       tokens,
                                       vocab_size_,
                                       stream);
            }
        }
    }

    void OutputCELoss(Data& data, BatchCopy& copy, const vector<shared_ptr<RequestCache>>& rs)
    {
        if (tp_rank_ != 0) {
            return;
        }
        for (const auto& [i, range] : data.ce_loss_segments) {
            if (auto& c = *rs[i]; !c.input_ce_loss) {
                copy(c.ce_loss, 1, c.req->outputs.at("ce_loss").buffer());
            }
        }
    }

    template<class Ranges>
    void OutputHiddenStates(const Ranges& ranges, const Tensor& h, int type, const vector<shared_ptr<RequestCache>>& rs)
    {
        for (const auto& [i, t, src, dst] : ranges) {
            if (t == type) {
                auto& out = rs[i]->req->outputs.at("last_hidden_state");
                if (tp_rank_ == 0) {
                    // dbg(&src, &dst);
                    Copy(h.slice(src.begin(), (int)src.size()), out.slice(dst.begin(), (int)dst.size()));
                }
            }
        }
    }

    void ComputeAndOutputLogits(Data&                                   data,  //
                                const Tensor&                           h,
                                BatchCopy&                              copy,
                                const vector<shared_ptr<RequestCache>>& rs)
    {
        const int step_size = max_logits_len_;

        // Coroutine frame
        int  p      = 0;
        auto ranges = data.output_logits;

        using Size = Interval::Size;

        bool success = ranges.empty();
        // Erode the range iteratively until empty
        for (auto r = data.full_logits | data.full_ce_loss; r; r = -step_size | r) {
            // dbg(&r);
            if (auto chunk = r & Interval{r.begin(), Size{step_size}}) {
                // dbg(&chunk);
                // Compute & output full logits by chunks
                // The chunked logits feeds two independent consumers
                auto logits = lm_head_(h.slice(chunk.begin(), (int)chunk.size()));
                if (!success) {
                    success = OutputLogitsImpl(ranges, p, logits, chunk.begin(), 2, rs);
                }
                ComputeCeLoss(data, logits, chunk.begin(), rs);
            }
        }

        TM_CHECK(success);  // every type-2 logits range must have been output

        // CE loss is fully accumulated now that the chunk loop is done; emit it.
        OutputCELoss(data, copy, rs);
    }

    template<class Ranges>
    void OutputLogits(Ranges& ranges_, const Tensor& l, int type, const vector<shared_ptr<RequestCache>>& rs)
    {
        // Coroutine frame
        int  p      = 0;
        auto ranges = ranges_;

        TM_CHECK(OutputLogitsImpl(ranges, p, l, /* base */ 0, type, rs));
    }

    template<class Ranges>
    bool OutputLogitsImpl(
        Ranges& ranges, int& p, const Tensor& l, int base, int type, const vector<shared_ptr<RequestCache>>& rs)
    {
        // dbg("OutputLogitsImpl");
        const auto stream = core::Context::stream().handle();
        for (; p < ranges.size(); ++p) {
            if (auto& [i, t, src, dst] = ranges[p]; t == type) {
                Tensor&        out   = rs[i]->req->outputs.at("logits");
                const DataType dtype = out.dtype();
                TM_CHECK_LE(base, src.begin());  // logical error
                if (Interval msrc = src & Interval{base, Interval::Size{(int)l.shape(0)}}) {
                    const int tokens = (int)msrc.size();
                    Interval  mdst{dst.begin(), msrc.size()};
                    // TODO: support strides in `DLTensor`, so that batched 1D copy can be used
                    if (tp_rank_ == 0) {
                        // dbg(&mdst, &msrc, tokens, out, base, l);
                        TM_CHECK_EQ(cudaMemcpy2DAsync(out.slice(mdst.begin(), tokens).raw_data(),
                                                      byte_size(dtype, out.stride(0)),
                                                      l.slice(msrc.begin() - base, tokens).raw_data(),
                                                      byte_size(dtype, l.stride(0)),
                                                      byte_size(dtype, vocab_size_),
                                                      tokens,
                                                      cudaMemcpyDefault,
                                                      stream),
                                    0);
                    }
                    // move to next request if they are empty after the erosion
                    src = -(int)msrc.size() | src;
                    dst = -(int)mdst.size() | dst;
                }
                // dbg(&src, (int)src.size(), &dst, (int)dst.size());
                if (src) {
                    // request not compeleted, suspend and wait for next chunk
                    return false;
                }
            }
        }
        return true;
    }

    void OutputHiddenStatesAndLogits(int phase, TensorMap& env, int type)
    {
        auto& d = data_.at(phase);
        auto& b = *env.at("batch").data<BatchData*>()[0];

        if (type == 2 && d.full_states) {
            auto hidden_states = env.consume("full_hidden_states");
            if (!d.output_states.empty()) {
                OutputHiddenStates(d.output_states, hidden_states, 2, b.rc);
            }
            if (d.full_logits || d.full_ce_loss) {
                auto& copy = *env.at("copy").data<BatchCopy*>()[0];
                ComputeAndOutputLogits(d, hidden_states, copy, b.rc);
            }
        }

        if (type == 1) {
            if (!d.output_states.empty()) {
                OutputHiddenStates(d.output_states, env.at("hidden_states"), 1, b.rc);
            }
            if (!d.output_logits.empty()) {
                OutputLogits(d.output_logits, env.at("logits"), 1, b.rc);
            }
        }
    }
};

OutputProcessor::~OutputProcessor() = default;

OutputProcessor::OutputProcessor(
    int vocab_size, int max_logits_len, int tp_rank, int phases, std::function<Tensor(const Tensor&)> lm_head):
    impl_{std::make_unique<Impl>(vocab_size, max_logits_len, tp_rank, phases, std::move(lm_head))}
{
}

void OutputProcessor::Run(BatchOp op, int phase, TensorMap& env)
{
    switch (op) {
        case BatchOp::kAdd:
            return impl_->Add(phase, env);
        case BatchOp::kSetup:
            return impl_->Setup(phase, env);
        case BatchOp::kPrepare:
            return impl_->Prepare(phase, env);
        default:
            return;
    }
}

void OutputProcessor::OutputHiddenStatesAndLogits(int phase, TensorMap& env, int type)
{
    return impl_->OutputHiddenStatesAndLogits(phase, env, type);
}

}  // namespace turbomind

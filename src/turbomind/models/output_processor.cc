
#include "src/turbomind/models/output_processor.h"

#include <functional>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/cross_entropy_kernels.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::shared_ptr;
using std::unique_ptr;

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

    struct OutputRange {
        std::shared_ptr<Request> request;
        int                      type;
        Interval                 src;
        Interval                 dst;
    };

    // A CE-loss scoring segment for one request in the current forward. Unlike `main`, which carries
    // the `RequestCache` on the batch and reaches `c.ce_loss`/`c.input_ce_loss` on the executor
    // thread, our executor side never touches `Sequence`. So we capture everything the executor needs
    // at Setup time: the request (for `outputs["ce_loss"]`), a handle to the Sequence's persistent
    // rank-0 accumulator, the hidden-buffer position interval, and whether `input_ce_loss` was fully
    // consumed this forward (chunked prefill emits only on the final forward).
    struct CeLossSegment {
        std::shared_ptr<Request> request;
        Buffer_<float>           ce_loss;
        Interval                 range;
        bool                     last;
    };

    struct Data {
        Interval full_states;  // requested range for full hidden states
        Interval full_logits;  // requested range for full logits

        vector<OutputRange> output_states;
        vector<OutputRange> output_logits;

        Interval full_ce_loss;  // requested range for CE-loss logits
        // Per CE scoring segment; `ce_targets` is indexed by the segment's hidden-buffer position
        // (the `range`), so no extra offset is stored.
        vector<CeLossSegment> ce_loss_segments;
        Buffer_<int>          ce_targets;
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
        const Buffer_<Sequence*> rc = env.at("requests").buffer();

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

        // const auto& rc = env.at("batch").data<BatchData*>()[0]->rc;
        Buffer_<Sequence*> rc   = env.at("requests").buffer();
        auto&              copy = *env.at("copy").data<BatchCopy*>()[0];

        vector<Interval> all_tokens;
        vector<Interval> sel_tokens;
        bool             has_ce = false;
        for (int i = 0; i < rc.size(); ++i) {
            using Size = Interval::Size;
            auto& c    = *rc[i];
            all_tokens.emplace_back(c.history_len + c.inflight_input_len, Size{c.input_len});
            sel_tokens.emplace_back(c.history_len + c.inflight_input_len + c.input_len - 1, Size{1});
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
                    d.output_states.push_back({c.req, type, m.src, m.dst});
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
                    d.output_logits.push_back({c.req, type, m.src, m.dst});
                }
            }
            if (c.input_ce_loss) {
                if (tp_rank_ == 0 && !c.ce_loss) {
                    // Per-request accumulator, allocated and zeroed once; persists across the
                    // chunked-prefill forwards that erode `input_ce_loss`.
                    c.ce_loss = {1, kDEVICE};
                    Clear(c.ce_loss);
                }
                Matching m{c.input_ce_loss, /* offset_d */ 0};
                if (m(all_tokens[i], offset, d.full_ce_loss)) {
                    if (tp_rank_ == 0) {
                        copy(c.token_ids + m.dst.begin() + 1, (int)m.src.size(), d.ce_targets.data() + m.src.begin());
                    }
                    // Capture everything the executor side needs (no `Sequence` access there): the
                    // request, a handle to its persistent accumulator, the hidden-buffer range, and
                    // whether this forward fully consumed `input_ce_loss` (emit only then).
                    d.ce_loss_segments.push_back({c.req, c.ce_loss, m.src, !c.input_ce_loss});
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

    template<class Ranges>
    void OutputHiddenStates(const Ranges& ranges, const Tensor& h, int type)
    {
        for (const auto& r : ranges) {
            if (r.type == type) {
                auto& out = r.request->outputs.at("last_hidden_state");
                if (tp_rank_ == 0) {
                    Copy(h.slice(r.src.begin(), (int)r.src.size()), out.slice(r.dst.begin(), (int)r.dst.size()));
                }
            }
        }
    }

    void ComputeCeLoss(Data& data, const Tensor& logits, int base)
    {
        if (tp_rank_ != 0 || data.ce_loss_segments.empty()) {
            return;
        }

        const auto     stream = core::Context::stream().handle();
        const Interval rows{base, Interval::Size{(int)logits.shape(0)}};

        for (auto& seg : data.ce_loss_segments) {
            if (auto src = seg.range & rows) {
                const int tokens        = (int)src.size();
                const int target_offset = src.begin();
                const int logit_offset  = src.begin() - base;
                invokeCrossEntropyLoss(seg.ce_loss.data(),
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

    void OutputCELoss(const Data& data)
    {
        if (tp_rank_ != 0) {
            return;
        }
        for (const auto& seg : data.ce_loss_segments) {
            if (seg.last) {  // input_ce_loss fully consumed -> accumulator is final
                Copy(seg.ce_loss, seg.request->outputs.at("ce_loss").buffer());
            }
        }
    }

    void ComputeAndOutputLogits(Data& data, const Tensor& h)
    {
        const int step_size = max_logits_len_;

        // Coroutine frame
        int  p      = 0;
        auto ranges = data.output_logits;

        using Size = Interval::Size;

        bool success = ranges.empty();
        // Erode the range iteratively until empty. Each chunk feeds two independent consumers:
        // full-logits output and CE-loss accumulation.
        for (auto r = data.full_logits | data.full_ce_loss; r; r = -step_size | r) {
            // dbg(&r);
            if (auto chunk = r & Interval{r.begin(), Size{step_size}}) {
                // dbg(&chunk);
                // Compute full logits by chunks
                auto logits = lm_head_(h.slice(chunk.begin(), (int)chunk.size()));
                if (!success) {
                    success = OutputLogitsImpl(ranges, p, logits, chunk.begin(), 2);
                }
                ComputeCeLoss(data, logits, chunk.begin());
            }
        }

        TM_CHECK(success);  // every type-2 logits range must have been output

        // CE loss is fully accumulated now that the chunk loop is done; emit it.
        OutputCELoss(data);
    }

    template<class Ranges>
    void OutputLogits(Ranges& ranges_, const Tensor& l, int type)
    {
        // Coroutine frame
        int  p      = 0;
        auto ranges = ranges_;

        TM_CHECK(OutputLogitsImpl(ranges, p, l, /* base */ 0, type));
    }

    template<class Ranges>
    bool OutputLogitsImpl(Ranges& ranges, int& p, const Tensor& l, int base, int type)
    {
        // dbg("OutputLogitsImpl");
        const auto stream = core::Context::stream().handle();
        for (; p < ranges.size(); ++p) {
            auto& r = ranges[p];
            if (r.type == type) {
                Tensor&        out   = r.request->outputs.at("logits");
                const DataType dtype = out.dtype();
                TM_CHECK_LE(base, r.src.begin());  // logical error
                if (Interval msrc = r.src & Interval{base, Interval::Size{(int)l.shape(0)}}) {
                    const int tokens = (int)msrc.size();
                    Interval  mdst{r.dst.begin(), msrc.size()};
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
                    r.src = -(int)msrc.size() | r.src;
                    r.dst = -(int)mdst.size() | r.dst;
                }
                // dbg(&r.src, (int)r.src.size(), &r.dst, (int)r.dst.size());
                if (r.src) {
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

        if (type == 2 && d.full_states) {
            auto hidden_states = env.consume("full_hidden_states");
            if (!d.output_states.empty()) {
                OutputHiddenStates(d.output_states, hidden_states, 2);
            }
            if (d.full_logits || d.full_ce_loss) {
                ComputeAndOutputLogits(d, hidden_states);
            }
        }

        if (type == 1) {
            if (!d.output_states.empty()) {
                OutputHiddenStates(d.output_states, env.at("hidden_states"), 1);
            }
            if (!d.output_logits.empty()) {
                OutputLogits(d.output_logits, env.at("logits"), 1);
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


#include "src/turbomind/models/input_processor.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/llama/SequenceManager.h"

namespace turbomind {

using std::vector;

struct InputProcessor::Impl {
public:
    Impl(const EngineParam& engine, const ModelParam& model, int phases):
        max_batch_size_{engine.max_batch_size}, max_forward_token_num_{engine.max_forward_token_num}
    {
        input_ids_buf_         = {max_forward_token_num_, kCPUpinned};
        input_ids_offsets_buf_ = {max_batch_size_ + 1, kCPUpinned};
        decode_token_pos_buf_  = {max_batch_size_, kCPUpinned};

        data_.reserve(phases);
        for (int i = 0; i < phases; ++i) {
            auto& d              = data_.emplace_back();
            d.input_ids          = empty_like(input_ids_buf_, kDEVICE);
            d.input_ids_offsets  = empty_like(input_ids_offsets_buf_, kDEVICE);
            d.selected_token_pos = empty_like(decode_token_pos_buf_, kDEVICE);

            d.autoreg_ids_pos = {max_batch_size_, kCPU};  // ! CPU buffer

            /// TODO: initialize only when required
            d.input_embeds_buf = {{max_forward_token_num_, (int)model.hidden_units}, model.data_type, kCPUpinned};
        }
    }

    int Add(RequestCache& c)
    {
        const auto& [r, s] = std::tie(*c.req, *c.seq);

        // trim input embeds
        if (!s.input_embeds_offsets.empty()) {
            Interval l{0, (int)s.tokens.size()};
            using Size    = Interval::Size;
            auto& embeds  = s.input_embeds;
            auto& offsets = s.input_embeds_offsets;
            int   i       = embeds.size() - 1;
            for (; i >= 0; --i) {
                Interval r{offsets[i], Size{(int)embeds[i].shape(0)}};
                if (auto o = r & l) {
                    if (o.end() < r.end()) {
                        embeds[i] = embeds[i].slice(0, o.end() - r.begin());
                    }
                    break;
                }
            }
            embeds.resize(i + 1);
            offsets.resize(i + 1);
        }

        if (auto ranges_ptr = r.inputs.try_("input_embedding_ranges")) {  // [n, 2]
            auto embeds = r.inputs.at("input_embeddings");                // [k, d]
            if (ranges_ptr->ndim() != 2 || embeds.ndim() != 2 || ranges_ptr->shape(1) != 2) {
                /// TODO: reject for invalid shapes
                return Request::kInvalid;
            }

            // clone the embeds if the request persists
            if (!r.session.end_flag) {
                auto tmp = std::exchange(embeds, empty_like(embeds));
                std::copy_n((const uint8_t*)tmp.raw_data(), tmp.byte_size(), (uint8_t*)embeds.raw_data());
            }

            const auto [sum, dim] = embeds.shapes(0, 1);
            const auto n          = ranges_ptr->shape(0);
            const auto ranges     = ranges_ptr->data<int>();

            int offset = 0;
            int last   = c.step0;
            for (int i = 0; i < n; ++i) {
                Interval range{c.step0 + ranges[i * 2], c.step0 + ranges[i * 2 + 1]};
                auto     size = (int)range.size();
                if (range.begin() < last) {
                    /// TODO: reject for non-sorted ranges
                    return Request::kInvalid;
                }
                if (range.end() > c.seq_len) {
                    /// TODO: reject for dst range OOB
                    return Request::kInvalid;
                }
                if (offset + size > sum) {
                    /// TODO: reject for src range OOB
                    return Request::kInvalid;
                }
                s.input_embeds_offsets.push_back(range.begin());
                s.input_embeds.push_back(embeds.slice(offset, size));  // reference into `embeds`
                offset += size;
                last = range.end();
            }
        }

        return 0;
    }

    void Add(int phase, TensorMap& env)
    {
        const Buffer_<RequestCache*> rc = env.at("requests").buffer();
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *TM_CHECK_NOTNULL(rc[i]);
            if (c.status == 0) {
                c.status = Add(c);
            }
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        auto& d    = data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        const auto& rc = b.rc;

        input_ids_offsets_buf_[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            input_ids_offsets_buf_[i + 1] = input_ids_offsets_buf_[i];
            if (const auto& c = *rc[i]; TM_UNLIKELY(!c.autoregres)) {
                const auto src = c.token_ids + c.history_len + c.alpha;
                std::copy_n(src, c.input_len, input_ids_buf_.data() + input_ids_offsets_buf_[i]);
                // dbg(std::vector<int>(src, src + c.input_len));
                d.autoreg_ids_pos[i] = -1;
                input_ids_offsets_buf_[i + 1] += c.input_len;
            }
            else {
                d.autoreg_ids_pos[i] = input_ids_offsets_buf_[i];
                input_ids_offsets_buf_[i + 1] += 1;
            }
            decode_token_pos_buf_[i] = input_ids_offsets_buf_[i + 1] - 1;
        }

        // dbg(core::to_vector<int>(input_ids_offsets_buf_.slice(0, bsz + 1)));
        // dbg(core::to_vector<int>(decode_token_pos_buf_.slice(0, bsz)));

        copy(input_ids_buf_, input_ids_offsets_buf_[b.bsz], d.input_ids);
        copy(decode_token_pos_buf_, b.bsz, d.selected_token_pos);
        copy(input_ids_offsets_buf_, b.bsz + 1, d.input_ids_offsets);

        // dbg(decode_token_pos_buf_[0]);

        d.input_token_num = input_ids_offsets_buf_[b.bsz];
        // dbg(d.input_token_num);

        env.produce("token_num", Buffer{&d.input_token_num, 1, kCPU});

        ////////////////////////////////////////////////////////////////
        /// input embeddings
        d.input_embeds_coords.clear();
        auto embed_ptr = (uint8_t*)d.input_embeds_buf.raw_data();
        for (int k = 0; k < rc.size(); ++k) {
            if (auto& c = *rc[k]; !c.autoregres) {
                const auto& embeds  = c.seq->input_embeds;
                const auto& offsets = c.seq->input_embeds_offsets;
                Interval    p{input_ids_offsets_buf_[k], input_ids_offsets_buf_[k + 1]};
                Interval    s{c.history_len + c.alpha, p.size()};
                for (int i = (int)offsets.size() - 1; i >= 0; --i) {
                    Interval r{offsets[i], Interval::Size{(int)embeds[i].shape(0)}};
                    auto     o = r & s;
                    if (auto size = (int)o.size()) {
                        auto src  = embeds[i].slice(o.begin() - r.begin(), size);
                        embed_ptr = std::copy_n((const uint8_t*)src.raw_data(), src.byte_size(), embed_ptr);
                        d.input_embeds_coords.emplace_back(size, p.begin() + (o.begin() - s.begin()));
                    }
                }
            }
        }
    }

    void Prepare(int phase, TensorMap& env)
    {
        auto& d    = data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        // last output token + draft tokens
        const Buffer_<int> autoreg_ids = env.at("autoreg_ids").buffer();

        // core::CopyT copy{};

        if (auto g = copy.group()) {
            for (int i = 0; i < b.bsz; ++i) {
                if (auto pos = d.autoreg_ids_pos[i]; pos >= 0) {
                    TM_CHECK_LT(b.perm[i], b.bs0);
                    copy(autoreg_ids.data() + b.perm[i], 1, &d.input_ids[pos]);
                }
            }
        }

        env.produce("input_ids", d.input_ids.slice(0, d.input_token_num));
        env.produce("q_offsets", d.input_ids_offsets.slice(0, b.bsz + 1));
        env.produce("selected_token_pos", d.selected_token_pos.slice(0, b.bsz));
    }

    void PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy)
    {
        auto&      d           = data_.at(phase);
        const auto byte_stride = byte_size(embeds.dtype(), embeds.stride(0));
        int        offset      = 0;
        for (const auto& [size, pos] : d.input_embeds_coords) {
            auto src = d.input_embeds_buf.slice(offset, size);
            copy((uint8_t*)src.raw_data(), src.byte_size(), (uint8_t*)embeds.raw_data() + byte_stride * pos);
            offset += size;
        }
    }

private:
    struct Data {
        Buffer_<int> input_ids;
        Buffer_<int> input_ids_offsets;
        int          input_token_num;

        Buffer_<int> selected_token_pos;

        Buffer_<int> autoreg_ids_pos;

        Tensor                      input_embeds_buf;
        vector<std::pair<int, int>> input_embeds_coords;  // (size, pos)
    };

private:
    const int max_batch_size_;
    const int max_forward_token_num_;

    vector<Data> data_;

    Buffer_<int> input_ids_buf_;
    Buffer_<int> input_ids_offsets_buf_;

    Buffer_<int> decode_token_pos_buf_;
};

InputProcessor::~InputProcessor() = default;

InputProcessor::InputProcessor(const EngineParam& engine, const ModelParam& model, int phases):
    impl_{std::make_unique<Impl>(engine, model, phases)}
{
}

void InputProcessor::Run(BatchOp op, int phase, TensorMap& env)
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

void InputProcessor::PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy)
{
    impl_->PatchEmbedding(phase, embeds, copy);
}

}  // namespace turbomind

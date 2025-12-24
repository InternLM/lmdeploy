

#include "src/turbomind/generation/stop_criteria.h"
#include "src/turbomind/generation/utils.h"

#include "src/turbomind/kernels/stop_criteria_kernels.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

namespace turbomind {

struct StopCriteriaData {
    explicit StopCriteriaData(int batch_size)
    {
        stop_words  = {batch_size * 2 * kMaxStopBadWordsLen, kDEVICE};
        max_seq_len = {batch_size, kDEVICE};
    }
    Buffer_<int> stop_words;
    Buffer_<int> max_seq_len;
    Tensor_<int> stop_words_ten;  // reference int `stop_words`
};

StopCriteria::StopCriteria(const BaseGenerationParam& base, int phases): BaseGenerationParam{base}
{
    stop_words_buf_  = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kCPUpinned};
    max_seq_len_buf_ = {max_batch_size_, kCPUpinned};
    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<StopCriteriaData>(max_batch_size_));
    }
}

void StopCriteria::Setup(int phase, TensorMap& env)
{
    auto& d = *data_.at(phase);

    const auto& rs   = env.at("batch").data<BatchData*>()[0]->rc;
    auto&       copy = *env.at("copy").data<BatchCopy*>()[0];

    for (int i = 0; i < rs.size(); ++i) {
        max_seq_len_buf_[i] = rs[i]->max_seq_len;
    }
    copy(max_seq_len_buf_, rs.size(), d.max_seq_len);

    d.stop_words_ten = {};
    init_stop_bad_words(&GenerationConfig::stop_ids,  //
                        "stop_words",
                        rs,
                        stop_words_buf_.data(),
                        d.stop_words.data(),
                        d.stop_words_ten,
                        copy);
}

void StopCriteria::Forward(int phase, TensorMap& env)
{
    auto& d = *data_.at(phase);

    const Buffer_<int*> token_ids_ptrs  = env.at("token_ids_ptrs").buffer();
    const Buffer_<int>  sequence_length = env.at("sequence_length").buffer();

    Buffer_<bool> finished = env.at("finished").buffer();

    const int batch_size = token_ids_ptrs.size();

    auto stream = core::Context::stream().handle();

    if (auto& stop_words = d.stop_words_ten) {
        TM_CHECK_EQ(stop_words.ndim(), 3);  // [batch, 2, len]
        size_t stop_words_len = stop_words.shape(2);
        invokeStopWordsCriterion_v2((const int**)token_ids_ptrs.data(),
                                    sequence_length.data(),
                                    stop_words.data(),
                                    finished.data(),
                                    stop_words_len,
                                    batch_size,
                                    stream);
        sync_check_cuda_error();
    }

    invokeLengthCriterion_v2(finished.data(),  //
                             sequence_length.data(),
                             d.max_seq_len.data(),
                             batch_size,
                             stream);
    sync_check_cuda_error();
}

}  // namespace turbomind

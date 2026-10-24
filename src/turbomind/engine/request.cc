

#include "src/turbomind/engine/request.h"

#include <iterator>

namespace turbomind {

namespace {

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, ", "));
    if (!vec.empty()) {
        os.seekp(-2, std::ios_base::end);
    }
    os << "]";
    return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const GenerationConfig& c)
{
    os << "GenerationConfig { ";
    os << "max_new_tokens=" << c.max_new_tokens;
    os << ", min_new_tokens=" << c.min_new_tokens;
    os << ", eos_ids=" << c.eos_ids;
    os << ", stop_ids=[" << c.stop_ids[0] << ", " << c.stop_ids[1] << "]";
    os << ", bad_ids=[" << c.bad_ids[0] << ", " << c.bad_ids[1] << "]";
    os << ", top_p=" << c.top_p;
    os << ", top_k=" << c.top_k;
    os << ", min_p=" << c.min_p;
    os << ", temperature=" << c.temperature;
    os << ", repetition_penalty=" << c.repetition_penalty;
    os << ", random_seed=" << c.random_seed;
    os << ", output_logprobs=" << c.output_logprobs;
    os << ", return_ppl=" << c.return_ppl;
    os << ", output_hidden_states=" << c.output_last_hidden_state;
    os << ", output_logits=" << c.output_logits;
    os << " }";
    return os;
}

void UpdateState(Request& request, RequestState state)
{
    try {
        auto next     = new RequestState{std::move(state)};
        auto previous = request.state->exchange(next);
        if (!previous && request.forward_cb) {
            request.forward_cb();
        }
    }
    catch (const std::exception& e) {
        TM_LOG_ERROR("Error invoking callback for ({}): {}", request.id, e.what());
    }
    catch (...) {
        TM_LOG_ERROR("Unknown error invoking callback for ({})", request.id);
    }
}

std::function<void()> MakeRequestSignal(std::shared_ptr<Request> request, int status, int seq_len)
{
    RequestState state;
    state.status  = status;
    state.seq_len = seq_len;

    if (request->metrics) {
        auto&            metrics = *request->metrics;
        std::scoped_lock lock(metrics.spec_mutex);

        if (!metrics.num_accepted_tokens_per_pos.empty()) {
            state.num_drafts                  = metrics.num_drafts;
            state.num_draft_tokens            = metrics.num_draft_tokens;
            state.num_accepted_tokens         = metrics.num_accepted_tokens;
            state.num_accepted_tokens_per_pos = metrics.num_accepted_tokens_per_pos;
        }
    }

    return
        [request = std::move(request), state = std::move(state)]() mutable { UpdateState(*request, std::move(state)); };
}

}  // namespace turbomind

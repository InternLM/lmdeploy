

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
    os << ", output_hidden_states=" << c.output_last_hidden_state;
    os << ", output_logits=" << c.output_logits;
    os << " }";
    return os;
}

void UpdateState(Request& r, int status, int seq_len)
{
    try {
        auto new_state = new RequestState{status, seq_len};
        auto old_state = r.state->exchange(new_state);
        if (!old_state && r.forward_cb) {
            r.forward_cb();
        }
    }
    catch (const std::exception& e) {
        TM_LOG_ERROR("Error invoking callback for (%lu): %s", r.id, e.what());
    }
    catch (...) {
        TM_LOG_ERROR("Unknown error invoking callback for (%lu)", r.id);
    }
}

}  // namespace turbomind

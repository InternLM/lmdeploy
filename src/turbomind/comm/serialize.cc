// Copyright (c) OpenMMLab. All rights reserved.

#include <sstream>
#include <vector>

#include "src/turbomind/comm/serialize.h"
#include "src/turbomind/engine/request.h"

namespace turbomind::comm {

std::vector<char> streambuf_to_vector(std::streambuf* sb)
{
    auto start = sb->pubseekoff(0, std::ios::beg, std::ios::in);
    auto end   = sb->pubseekoff(0, std::ios::end, std::ios::in);
    auto size  = end - start;

    std::vector<char> buffer(size);
    sb->pubseekpos(start);
    sb->sgetn(buffer.data(), size);
    return buffer;
}

void serialize(std::ostream& os, const std::string& s)
{
    int size = s.length();
    serialize(os, size);
    os << s;
}

void deserialize(std::istream& is, std::string& s)
{
    int size;
    deserialize(is, size);
    s.resize(size);
    is.read(s.data(), size);
}

void serialize(std::ostream& os, const GenerationConfig& gen)
{
    serialize(os, gen.max_new_tokens);
    serialize(os, gen.min_new_tokens);
    serialize(os, gen.eos_ids);
    serialize(os, gen.stop_ids[0]);
    serialize(os, gen.stop_ids[1]);
    serialize(os, gen.bad_ids[0]);
    serialize(os, gen.bad_ids[1]);
    serialize(os, gen.top_k);
    serialize(os, gen.top_p);
    serialize(os, gen.min_p);
    serialize(os, gen.temperature);
    serialize(os, gen.repetition_penalty);
    serialize(os, gen.random_seed);
    serialize(os, gen.output_logprobs);
    serialize(os, gen.output_last_hidden_state);
    serialize(os, gen.output_logits);
}

void deserialize(std::istream& is, GenerationConfig& gen)
{
    deserialize(is, gen.max_new_tokens);
    deserialize(is, gen.min_new_tokens);
    deserialize(is, gen.eos_ids);
    deserialize(is, gen.stop_ids[0]);
    deserialize(is, gen.stop_ids[1]);
    deserialize(is, gen.bad_ids[0]);
    deserialize(is, gen.bad_ids[1]);
    deserialize(is, gen.top_k);
    deserialize(is, gen.top_p);
    deserialize(is, gen.min_p);
    deserialize(is, gen.temperature);
    deserialize(is, gen.repetition_penalty);
    deserialize(is, gen.random_seed);
    deserialize(is, gen.output_logprobs);
    deserialize(is, gen.output_last_hidden_state);
    deserialize(is, gen.output_logits);
}

void serialize(std::ostream& os, const SessionParam& sess)
{
    serialize(os, sess.id);
    serialize(os, sess.step);
    serialize(os, sess.start_flag);
    serialize(os, sess.end_flag);
    serialize(os, sess.kill_flag);
}

void deserialize(std::istream& is, SessionParam& sess)
{
    deserialize(is, sess.id);
    deserialize(is, sess.step);
    deserialize(is, sess.start_flag);
    deserialize(is, sess.end_flag);
    deserialize(is, sess.kill_flag);
}

void serialize(std::ostream& os, const Layout& layout)
{
    serialize(os, layout.shape());
    serialize(os, layout.stride());
}

void deserialize(std::istream& is, Layout& layout)
{
    std::vector<ssize_t> shape;
    std::vector<ssize_t> stride;
    deserialize(is, shape);
    deserialize(is, stride);
    layout = Layout(std::move(shape), std::move(stride));
}

void serialize(std::ostream& os, const Buffer& buffer) {
    FT_CHECK(buffer.device() == turbomind::core::Device(kCPU));
    serialize(os, buffer.size());
    serialize(os, buffer.dtype());
    os.write((char*)buffer.raw_data(), buffer.byte_size());
}

void deserialize(std::istream& is, Buffer& buffer) {
    ssize_t size;
    DataType dtype;
    deserialize(is, size);
    deserialize(is, dtype);
    buffer = Buffer(size, dtype, turbomind::core::Device(kCPU));
    is.read((char*)buffer.raw_data(), buffer.byte_size());
}

void serialize(std::ostream& os, const Tensor& tensor)
{
    FT_CHECK(tensor.is_contiguous());
    serialize(os, tensor.layout());
    serialize(os, tensor.buffer());
}

void deserialize(std::istream& is, Tensor& tensor)
{
    Layout layout;
    Buffer buffer;
    deserialize(is, layout);
    deserialize(is, buffer);
    tensor = Tensor(std::move(buffer), std::move(layout));
}

void serialize(std::ostream& os, const TensorMap& map)
{
    int size = map.size();
    serialize(os, size);
    for (const auto& [key, tensor] : map) {
        serialize(os, key);
        serialize(os, tensor);
    }
}

void deserialize(std::istream& is, TensorMap& map)
{
    int size;
    deserialize(is, size);
    for (int i = 0; i < size; ++i) {
        std::string key;
        deserialize(is, key);
        Tensor tensor;
        deserialize(is, tensor);
        map.emplace(key, tensor);
    }
}

void serialize(std::ostream& os, const Request& req)
{
    serialize(os, req.id);
    serialize(os, req.unique_id);
    serialize(os, req.session);
    serialize(os, req.gen_cfg);
    serialize(os, req.stream_output);
    serialize(os, req.inputs);
    serialize(os, req.outputs);
    serialize(os, req.ec);
}

void deserialize(std::istream& is, Request& req)
{
    deserialize(is, req.id);
    deserialize(is, req.unique_id);
    deserialize(is, req.session);
    deserialize(is, req.gen_cfg);
    deserialize(is, req.stream_output);
    deserialize(is, req.inputs);
    deserialize(is, req.outputs);
    deserialize(is, req.ec);

    req.output_ids      = req.outputs.at("output_ids");
    req.sequence_length = req.outputs.at("sequence_length");
}

}  // namespace turbomind::comm

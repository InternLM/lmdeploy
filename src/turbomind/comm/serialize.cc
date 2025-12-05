// Copyright (c) OpenMMLab. All rights reserved.

#include <sstream>
#include <vector>

#include "src/turbomind/comm/serialize.h"
#include "src/turbomind/engine/request.h"

namespace turbomind::comm {

char* serialize(char* data, size_t& size, const std::string& s)
{
    int n = s.length();
    data  = serialize(data, size, n);
    if (data) {
        std::memcpy(data, s.data(), n);
    }
    size += n;
    return data ? data + n : data;
}

char* deserialize(std::string& s, char* data)
{
    int n;
    data = deserialize(n, data);
    s.resize(n);
    std::memcpy(s.data(), data, n);
    return data + n;
}

char* serialize(char* data, size_t& size, const GenerationConfig& gen)
{
    data = serialize(data, size, gen.max_new_tokens);
    data = serialize(data, size, gen.min_new_tokens);
    data = serialize(data, size, gen.eos_ids);
    data = serialize(data, size, gen.stop_ids[0]);
    data = serialize(data, size, gen.stop_ids[1]);
    data = serialize(data, size, gen.bad_ids[0]);
    data = serialize(data, size, gen.bad_ids[1]);
    data = serialize(data, size, gen.top_k);
    data = serialize(data, size, gen.top_p);
    data = serialize(data, size, gen.min_p);
    data = serialize(data, size, gen.temperature);
    data = serialize(data, size, gen.repetition_penalty);
    data = serialize(data, size, gen.random_seed);
    data = serialize(data, size, gen.output_logprobs);
    data = serialize(data, size, gen.output_last_hidden_state);
    data = serialize(data, size, gen.output_logits);
    return data;
}

char* deserialize(GenerationConfig& gen, char* data)
{
    data = deserialize(gen.max_new_tokens, data);
    data = deserialize(gen.min_new_tokens, data);
    data = deserialize(gen.eos_ids, data);
    data = deserialize(gen.stop_ids[0], data);
    data = deserialize(gen.stop_ids[1], data);
    data = deserialize(gen.bad_ids[0], data);
    data = deserialize(gen.bad_ids[1], data);
    data = deserialize(gen.top_k, data);
    data = deserialize(gen.top_p, data);
    data = deserialize(gen.min_p, data);
    data = deserialize(gen.temperature, data);
    data = deserialize(gen.repetition_penalty, data);
    data = deserialize(gen.random_seed, data);
    data = deserialize(gen.output_logprobs, data);
    data = deserialize(gen.output_last_hidden_state, data);
    data = deserialize(gen.output_logits, data);
    return data;
}

char* serialize(char* data, size_t& size, const SessionParam& sess)
{
    data = serialize(data, size, sess.id);
    data = serialize(data, size, sess.step);
    data = serialize(data, size, sess.start_flag);
    data = serialize(data, size, sess.end_flag);
    data = serialize(data, size, sess.kill_flag);
    return data;
}

char* deserialize(SessionParam& sess, char* data)
{
    data = deserialize(sess.id, data);
    data = deserialize(sess.step, data);
    data = deserialize(sess.start_flag, data);
    data = deserialize(sess.end_flag, data);
    data = deserialize(sess.kill_flag, data);
    return data;
}

char* serialize(char* data, size_t& size, const Layout& layout)
{
    data = serialize(data, size, layout.shape());
    data = serialize(data, size, layout.stride());
    return data;
}

char* deserialize(Layout& layout, char* data)
{
    std::vector<ssize_t> shape;
    std::vector<ssize_t> stride;
    data   = deserialize(shape, data);
    data   = deserialize(stride, data);
    layout = Layout(std::move(shape), std::move(stride));
    return data;
}

char* serialize(char* data, size_t& size, const Buffer& buffer)
{
    FT_CHECK(buffer.device() == turbomind::core::Device(kCPU));
    data = serialize(data, size, buffer.size());
    data = serialize(data, size, buffer.dtype());
    if (data) {
        std::memcpy(data, buffer.raw_data(), buffer.byte_size());
    }
    size += buffer.byte_size();
    return data ? data + buffer.byte_size() : data;
}

char* deserialize(Buffer& buffer, char* data)
{
    ssize_t  size;
    DataType dtype;
    data   = deserialize(size, data);
    data   = deserialize(dtype, data);
    buffer = Buffer(size, dtype, turbomind::core::Device(kCPU));
    std::memcpy(buffer.raw_data(), data, buffer.byte_size());
    return data + buffer.byte_size();
}

char* serialize(char* data, size_t& size, const Tensor& tensor)
{
    FT_CHECK(tensor.is_contiguous());
    data = serialize(data, size, tensor.layout());
    data = serialize(data, size, tensor.buffer());
    return data;
}

char* deserialize(Tensor& tensor, char* data)
{
    Layout layout;
    Buffer buffer;
    data   = deserialize(layout, data);
    data   = deserialize(buffer, data);
    tensor = Tensor(std::move(buffer), std::move(layout));
    return data;
}

char* serialize(char* data, size_t& size, const TensorMap& map)
{
    data = serialize(data, size, (int)map.size());
    for (const auto& [key, tensor] : map) {
        data = serialize(data, size, key);
        data = serialize(data, size, tensor);
    }
    return data;
}

char* deserialize(TensorMap& map, char* data)
{
    int size;
    data = deserialize(size, data);
    for (int i = 0; i < size; ++i) {
        std::string key;
        data = deserialize(key, data);
        Tensor tensor;
        data = deserialize(tensor, data);
        map.emplace(std::move(key), std::move(tensor));
    }
    return data;
}

char* serialize(char* data, size_t& size, const Request& req)
{
    // TODO: support grammar
    data = serialize(data, size, req.id);
    data = serialize(data, size, req.unique_id);  // consider dp ?
    data = serialize(data, size, req.session);
    data = serialize(data, size, req.gen_cfg);
    data = serialize(data, size, req.stream_output);
    data = serialize(data, size, req.inputs);
    data = serialize(data, size, req.outputs);
    data = serialize(data, size, req.ec);
    return data;
}

char* deserialize(Request& req, char* data)
{
    data = deserialize(req.id, data);
    data = deserialize(req.unique_id, data);
    data = deserialize(req.session, data);
    data = deserialize(req.gen_cfg, data);
    data = deserialize(req.stream_output, data);
    data = deserialize(req.inputs, data);
    data = deserialize(req.outputs, data);
    data = deserialize(req.ec, data);

    req.output_ids      = req.outputs.at("output_ids");
    req.sequence_length = req.outputs.at("sequence_length");
    return data;
}

}  // namespace turbomind::comm

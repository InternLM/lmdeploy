#pragma once

namespace turbomind {
namespace multimodal {

enum class Modality
{
    kImage,
    kVideo,
    kAudio,
    kTimeSeries,
};

struct Input {
    virtual ~Input() = default;
};

}  // namespace multimodal
}  // namespace turbomind

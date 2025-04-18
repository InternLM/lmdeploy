
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

/// TODO: remove this dependency
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::core {

class Allocator;
class Buffer;
class Stream;
class Event;
class Context;

using std::shared_ptr;
using std::vector;

using ssize_t = std::ptrdiff_t;

}  // namespace turbomind::core

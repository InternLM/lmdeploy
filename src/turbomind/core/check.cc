
#include <cstdlib>
#include <filesystem>
#include <sstream>

#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind::core {

namespace {

std::string StripSrcPrefix(const char* file)
{
    static const char* flag = std::getenv("TM_SRC_FULL_PATH");
    if (flag) {
        return file;
    }

    std::filesystem::path path{file};
    std::filesystem::path ret{path};  // return the original path if anchor is not found

    constexpr auto anchor = "turbomind";

    bool found = false;

    for (const auto& x : path) {
        if (x == anchor) {
            found = true;
            ret.clear();
        }
        else if (found) {
            ret /= x;
        }
    }

    return ret;
}

}  // namespace

CheckOpStringBuilder::CheckOpStringBuilder()
{
    oss_ = new std::ostringstream;
}

std::ostream* CheckOpStringBuilder::ForVal1()
{
    (*oss_) << "(";
    return oss_;
}
std::ostream* CheckOpStringBuilder::ForVal2()
{
    (*oss_) << " vs. ";
    return oss_;
}
std::string* CheckOpStringBuilder::NewString()
{
    (*oss_) << ")";
    return new std::string{oss_->str()};
}

CheckErrorStream::CheckErrorStream(const char* file, int line, const char* expr)
{
    oss_ = new std::ostringstream{};
    *oss_ << StripSrcPrefix(file) << "(" << line << "): Check failed: " << expr << " ";
}

CheckErrorStream::CheckErrorStream(const char* file, int line, const char* expr, std::string* str):
    CheckErrorStream{file, line, expr}
{
    *oss_ << *str << " ";
}

void CheckErrorStream::Report()
{
    TM_LOG_ERROR(oss_->str().c_str());
    std::abort();
}

void ReportNullError(const char* file, int line, const char* expr)
{
    TM_LOG_ERROR("%s(%d): '%s' Must be non NULL", StripSrcPrefix(file).c_str(), line, expr);
    std::abort();
}

}  // namespace turbomind::core
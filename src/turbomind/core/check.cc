
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"

namespace turbomind::core {

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

CheckErrorStream::CheckErrorStream(const char* file, int line, const char* expr): file_{file}, line_{line}
{
    oss_ = new std::ostringstream{};
    // *oss_ << StripSrcPrefix(file) << "(" << line << "): Check failed: " << expr << " ";
    *oss_ << "Check failed: " << expr << " ";
}

CheckErrorStream::CheckErrorStream(const char* file, int line, const char* expr, std::string* str):
    CheckErrorStream{file, line, expr}
{
    *oss_ << *str << " ";
}

void CheckErrorStream::Report()
{
    Scope _("TM_CHECK", file_, line_);
    Logger::Instance().LogFatalImpl(file_, line_, oss_->str());
}

void ReportNullError(const char* file, int line, const char* expr)
{
    Scope _("TM_CHECK_NOTNULL", file, line);
    Logger::Instance().LogFatalImpl(file, line, fmt::format("'{}' Must be non NULL", expr));
}

}  // namespace turbomind::core

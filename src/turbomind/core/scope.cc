#include "src/turbomind/core/scope.h"

#include <cstring>

namespace turbomind::core {
namespace {

// Position of `close` matching the `open` at `start` (scan forward).
size_t match_fwd(const std::string& s, size_t start, char open, char close)
{
    for (int d = 0; start < s.size(); ++start) {
        if (s[start] == open)
            ++d;
        else if (s[start] == close && --d == 0)
            return start;
    }
    return std::string::npos;
}

// Position of `open` matching the `close` just before `end` (scan backward).
size_t match_rev(const std::string& s, size_t end, char close, char open)
{
    for (int d = 0; end > 0; --end) {
        if (s[end - 1] == close)
            ++d;
        else if (s[end - 1] == open && --d == 0)
            return end - 1;
    }
    return std::string::npos;
}

}  // namespace

Scope::Scope(const char* name, const char* file, int line)
{
    Context::push_scope({name, file, line});
}

Scope::Scope(const char* name, const char* file, int line, scope_type type)
{
    Context::push_scope({name, file, line, type});
}

Scope::~Scope()
{
    Context::pop_scope();
}

std::string Scope::trace()
{
    return Context::scope_trace();
}

int Scope::depth()
{
    return Context::scope_depth();
}

std::string StripFunctionSignature(std::string_view name)
{
    std::string s{name};

    // 1. Strip trailing " [with <...>]" (GCC/Clang template args)
    if (auto p = s.rfind(" [with "); p != std::string::npos)
        if (match_fwd(s, p + 1, '[', ']') != std::string::npos)
            s.resize(p);

    // 2. Strip trailing member-function qualifiers
    for (int again = 1; again;) {
        again = 0;
        for (auto q : {" const", " volatile"}) {
            auto n = std::strlen(q);
            if (s.size() >= n && !s.compare(s.size() - n, n, q))
                s.resize(s.size() - n), again = 1;
        }
    }

    // 3. Replace parameter list with "()"
    if (!s.empty() && s.back() == ')')
        if (auto p = match_rev(s, s.size(), ')', '('); p != std::string::npos)
            s.replace(p, s.size() - p, "()");

    // 4. Strip function template "<...>" immediately before "()"
    //    (class template params like Container<T>::method are before "::", kept)
    if (s.size() >= 4 && s[s.size() - 3] == '>')
        if (auto p = match_rev(s, s.size() - 2, '>', '<'); p != std::string::npos)
            s.erase(p, s.size() - 2 - p);

    // 5. Strip return type: everything before last top-level space
    // clang-format off
    int a = 0, p = 0, b = 0;
    for (size_t i = s.size(); i > 0; --i) {
        switch (s[i - 1]) {
            case '>': ++a; break; case '<': --a; break;
            case ')': ++p; break; case '(': --p; break;
            case ']': ++b; break; case '[': --b; break;
            case ' ': if (!a && !p && !b) { s.erase(0, i); return s; }
        }
    }
    // clang-format on
    return s;
}

}  // namespace turbomind::core

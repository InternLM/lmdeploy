#include "src/turbomind/core/core.h"

#include "catch2/catch_test_macros.hpp"

using namespace turbomind;

TEST_CASE("test scope", "[scope]")
{
    using core::Context;
    using core::Scope;

    // Empty trace
    REQUIRE(Context::scope_depth() == 0);
    REQUIRE(Context::scope_trace().empty());

    {
        TM_SCOPE("outer");
        REQUIRE(Context::scope_depth() == 1);

        {
            TM_SCOPE("inner");
            REQUIRE(Context::scope_depth() == 2);

            auto trace = Context::scope_trace();
            REQUIRE(trace.find("outer") != std::string::npos);
            REQUIRE(trace.find("inner") != std::string::npos);
            REQUIRE(trace.find("test_scope.cc") != std::string::npos);
        }

        REQUIRE(Context::scope_depth() == 1);
    }

    REQUIRE(Context::scope_depth() == 0);
    REQUIRE(Context::scope_trace().empty());
}

namespace {

std::string last_trace;

void free_test_func()
{
    TM_FUNCTION_SCOPE();
    last_trace = core::Context::scope_trace();
}

struct TestStruct {
    void method()
    {
        TM_FUNCTION_SCOPE();
        last_trace = core::Context::scope_trace();
    }
};

template<typename T>
int template_test_func(T)
{
    TM_FUNCTION_SCOPE();
    last_trace = core::Context::scope_trace();
    return 0;
}

struct OperatorTest {
    void operator()(int x)
    {
        TM_FUNCTION_SCOPE();
        last_trace = core::Context::scope_trace();
    }
};

std::string scope_call_trace;
int         scope_call_depth;
int         scope_call_side_effect;

void scope_call_target(int x)
{
    scope_call_side_effect = x;
    scope_call_trace       = core::Context::scope_trace();
    scope_call_depth       = core::Context::scope_depth();
}

}  // namespace

TEST_CASE("test scope call", "[scope]")
{
    using core::Context;

    REQUIRE(Context::scope_depth() == 0);
    REQUIRE(Context::scope_trace().empty());

    SECTION("basic call with stringified expression")
    {
        scope_call_side_effect = 0;
        scope_call_trace.clear();
        scope_call_depth = 0;

        TM_SCOPE_CALL(scope_call_target(42));

        // The function call actually executed within the scope
        REQUIRE(scope_call_side_effect == 42);

        // The trace shows just the function name, not the full expression
        REQUIRE(scope_call_trace.find("scope_call_target(42)") == std::string::npos);
        REQUIRE(scope_call_trace.find("scope_call_target") != std::string::npos);
        REQUIRE(scope_call_trace.find("test_scope.cc") != std::string::npos);

        // The scope depth was 1 during the call
        REQUIRE(scope_call_depth == 1);

        // Scope is cleaned up after the call returns
        REQUIRE(Context::scope_depth() == 0);
        REQUIRE(Context::scope_trace().empty());
    }

    SECTION("nested inside TM_SCOPE")
    {
        scope_call_side_effect = 0;
        scope_call_trace.clear();
        scope_call_depth = 0;

        {
            TM_SCOPE("outer");
            REQUIRE(Context::scope_depth() == 1);

            TM_SCOPE_CALL(scope_call_target(99));

            // Function executed
            REQUIRE(scope_call_side_effect == 99);

            // During the call, depth was 2 (outer + scope_call)
            REQUIRE(scope_call_depth == 2);

            // Trace captured during the call contained both names
            REQUIRE(scope_call_trace.find("outer") != std::string::npos);
            REQUIRE(scope_call_trace.find("scope_call_target(99)") == std::string::npos);
            REQUIRE(scope_call_trace.find("scope_call_target") != std::string::npos);

            // After TM_SCOPE_CALL, depth returns to 1 (outer still active)
            REQUIRE(Context::scope_depth() == 1);
        }

        REQUIRE(Context::scope_depth() == 0);
        REQUIRE(Context::scope_trace().empty());
    }

    SECTION("multiple sequential calls")
    {
        scope_call_side_effect = 0;
        scope_call_trace.clear();
        scope_call_depth = 0;

        TM_SCOPE_CALL(scope_call_target(1));
        REQUIRE(scope_call_side_effect == 1);
        REQUIRE(scope_call_depth == 1);
        REQUIRE(Context::scope_depth() == 0);

        TM_SCOPE_CALL(scope_call_target(2));
        REQUIRE(scope_call_side_effect == 2);
        REQUIRE(scope_call_depth == 1);
        REQUIRE(Context::scope_depth() == 0);

        // Trace from the second call contains the second expression
        REQUIRE(scope_call_trace.find("scope_call_target(2)") == std::string::npos);
        REQUIRE(scope_call_trace.find("scope_call_target") != std::string::npos);
    }
}

TEST_CASE("test function scope formatting", "[scope]")
{
    using core::Context;

    REQUIRE(Context::scope_depth() == 0);

    SECTION("free function")
    {
        free_test_func();
        REQUIRE(last_trace.find("free_test_func()") != std::string::npos);
        REQUIRE(last_trace.find("void ") == std::string::npos);
    }

    SECTION("member function")
    {
        TestStruct{}.method();
        REQUIRE(last_trace.find("TestStruct::method()") != std::string::npos);
        REQUIRE(last_trace.find("void ") == std::string::npos);
    }

    SECTION("template function")
    {
        template_test_func(42);
        REQUIRE(last_trace.find("template_test_func()") != std::string::npos);
        REQUIRE(last_trace.find("[with") == std::string::npos);
        REQUIRE(last_trace.find("(T)") == std::string::npos);
        REQUIRE(last_trace.find("int ") == std::string::npos);
    }

    SECTION("operator()")
    {
        OperatorTest{}(42);
        REQUIRE(last_trace.find("OperatorTest::operator()()") != std::string::npos);
        REQUIRE(last_trace.find("(int") == std::string::npos);
        REQUIRE(last_trace.find("void ") == std::string::npos);
    }

    SECTION("MSVC __FUNCSIG__")
    {
        using core::Scope;
        using core::scope_type;

        // Free function template with __cdecl
        {
            Scope _(R"(void __cdecl print<char,int,const char*>(char,int,const char *))",
                    __FILE__,
                    __LINE__,
                    scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("print()") != std::string::npos);
            REQUIRE(trace.find("<char,int,const char*>") == std::string::npos);
            REQUIRE(trace.find("void ") == std::string::npos);
            REQUIRE(trace.find("__cdecl") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // Member function with __thiscall and trailing const
        {
            Scope _("void __thiscall MyClass::method(int) const", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("MyClass::method()") != std::string::npos);
            REQUIRE(trace.find("__thiscall") == std::string::npos);
            REQUIRE(trace.find(" const") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // __stdcall free function
        {
            Scope _("int __stdcall FreeFunc(float)", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("FreeFunc()") != std::string::npos);
            REQUIRE(trace.find("int ") == std::string::npos);
            REQUIRE(trace.find("__stdcall") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);
    }

    SECTION("template class member (GCC)")
    {
        using core::Scope;
        using core::scope_type;

        // Template class: Container<T>::process(int) [with T = int]
        {
            Scope _("void Container<T>::process(int) [with T = int]", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("Container<T>::process()") != std::string::npos);
            REQUIRE(trace.find("[with") == std::string::npos);
            REQUIRE(trace.find("(int)") == std::string::npos);
            REQUIRE(trace.find("void ") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // Nested template: Outer<std::vector<T>>::method(int) [with T = int]
        {
            Scope _("void Outer<std::vector<T>>::method(int) [with T = int]", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("Outer<std::vector<T>>::method()") != std::string::npos);
            REQUIRE(trace.find("[with") == std::string::npos);
            REQUIRE(trace.find("(int)") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // Template class with operator(): Foo<U>::operator()(int) [with U = float]
        {
            Scope _("void Foo<U>::operator()(int) [with U = float]", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("Foo<U>::operator()()") != std::string::npos);
            REQUIRE(trace.find("[with") == std::string::npos);
            REQUIRE(trace.find("(int)") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // Class template + function template member (GCC)
        {
            Scope _("void Container<T>::method(U) [with T = int; U = float]", __FILE__, __LINE__, scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("Container<T>::method()") != std::string::npos);
            REQUIRE(trace.find("[with") == std::string::npos);
            REQUIRE(trace.find("(U)") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);

        // Class template + function template member (MSVC)
        {
            Scope _("void __cdecl Container<T>::method<U>(U) [with T = int; U = float]",
                    __FILE__,
                    __LINE__,
                    scope_type::function);
            auto  trace = Context::scope_trace();
            REQUIRE(trace.find("Container<T>::method()") != std::string::npos);
            REQUIRE(trace.find("<U>") == std::string::npos);
            REQUIRE(trace.find("[with") == std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);
    }

    SECTION("named scope unchanged")
    {
        {
            TM_SCOPE("my_named_scope");
            auto trace = Context::scope_trace();
            REQUIRE(trace.find("my_named_scope") != std::string::npos);
        }
        REQUIRE(Context::scope_depth() == 0);
    }
}

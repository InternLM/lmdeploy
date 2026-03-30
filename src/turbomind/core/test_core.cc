
#include <numeric>

#include "src/turbomind/core/core.h"

#include "catch2/catch_test_macros.hpp"

using namespace turbomind;

TEST_CASE("test check", "[check]")
{
    int zero = 0;

    TM_CHECK(!zero);

    TM_CHECK_EQ(42, 42) << "Ok";
    TM_CHECK_NE(42, 24) << "Ok";
    TM_CHECK_GE(50, 42) << "Ok";
    TM_CHECK_GT(50, 42) << "Ok";
    TM_CHECK_LE(42, 50) << "Ok";
    TM_CHECK_LT(42, 50) << "Ok";

    if (0) {
        TM_CHECK(zero);
        TM_CHECK_EQ(42, 43) << "Not "
                            << "Ok";
    }

    int  x = 42;
    auto p = TM_CHECK_NOTNULL(&x);
    REQUIRE(p == &x);

    if (0) {
        int* y{};
        TM_CHECK_NOTNULL(y);
        TM_CHECK_NOTNULL(std::shared_ptr<void>{});
    }

    auto y = TM_CHECK_NOTNULL(std::make_shared<int>(42));
    REQUIRE(*y == 42);

    TM_CHECK(y);
}

TEST_CASE("test allocator", "[allocator]")
{

    using core::Allocator;
    using core::Stream;

    Allocator a;
    REQUIRE(!a);

    Allocator b{kCPU};
    REQUIRE(b);
    REQUIRE(a != b);
    REQUIRE(b->device() == kCPU);
    Stream s{};
    REQUIRE(!b->stream());

    // std::vector<int> v(1 << 20);
    // std::iota(v.begin(), v.end(), 0);

    // auto p = (int*)b->allocate(sizeof(int) * v.size());
    // std::iota(p, p + v.size(), 0);

    // REQUIRE(v == std::vector(p, p + v.size()));
}

TEST_CASE("test context", "[context]")
{
    using core::Context;
    using core::ContextGuard;
    using core::Stream;
    using core::Allocator;

    Stream s0 = Stream::create();

    ContextGuard g0{s0, Allocator{kCPU}};

    REQUIRE(Context::stream());
    REQUIRE(Context::stream() == s0);

    auto a0 = Context::host_alloc();

    {
        Allocator a1(Context::stream(), false);  // device allocator
        REQUIRE(a1->device().type == kDEVICE);

        ContextGuard g1{a1};

        REQUIRE(Context::stream() == s0);
        REQUIRE(Context::device_alloc() == a1);
        REQUIRE(Context::host_alloc() == a0);

        {
            ContextGuard g2{Stream::create(), Allocator(kDEVICE)};
            REQUIRE(Context::device_alloc() != a1);
            REQUIRE(Context::stream() != s0);
        }

        REQUIRE(Context::stream() == s0);
        REQUIRE(Context::device_alloc() == a1);
    }

    REQUIRE(Context::stream() == s0);
}

TEST_CASE("test basic buffer", "[buffer]")
{
    using core::Buffer;
    using core::Buffer_;
    using core::Allocator;

    Buffer a;
    REQUIRE(!a);

    Buffer b;
    REQUIRE(!b);
    REQUIRE(a == b);

    std::vector v{0, 1, 2, 3, 4, 5, 6, 7};

    SECTION("reference into v")
    {
        b = Buffer(v.data(), v.size(), kCPU);
        REQUIRE(b.data<int>() == v.data());
        REQUIRE(b.raw_data() == v.data());
    }
    SECTION("shared ownership")
    {
        auto x = std::shared_ptr<int[]>(new int[v.size()]);
        std::copy(v.begin(), v.end(), x.get());
        b = Buffer(x, v.size(), data_type_v<int>, kCPU);
        REQUIRE(b.data<int>() == x.get());
        REQUIRE(b.raw_data() == x.get());
    }
    SECTION("allocation")
    {
        Allocator alloc{kCPU};
        b = Buffer(v.size(), data_type_v<int>, alloc);
        std::copy(v.begin(), v.end(), b.data<int>());
    }

    REQUIRE(b);
    REQUIRE(b.size() == v.size());
    REQUIRE(b.dtype() == data_type_v<int>);
    REQUIRE(b.byte_size() == sizeof(int) * v.size());
    auto c = b;
    REQUIRE(c == b);
    REQUIRE(b == c);
    REQUIRE(a != b);
    REQUIRE(b != a);
    REQUIRE(std::vector(b.data<int>(), b.data<int>() + b.size()) == v);

    auto s = b.slice(3, 2);
    REQUIRE(s.size() == 2);
    REQUIRE(s.raw_data() == b.data<int>() + 3);

    Buffer_<int> x;
    Buffer_<int> y = Buffer{data_type_v<int>};

    Buffer z = Buffer_<int>(1024, kCPU);

    x = z;

    for (int i = 0; i < z.size(); ++i) {
        x[i] = i;
    }

    std::vector<int> ref(1024);
    std::iota(ref.begin(), ref.end(), 0);
    REQUIRE(std::vector(x.begin(), x.end()) == ref);

    Buffer e;
    REQUIRE(!e.data_or((void*)0));
    REQUIRE(!e.data_or<int>(nullptr));

    Buffer_<int> w;
    REQUIRE(!w.data_or(nullptr));
    REQUIRE(!std::as_const(w).data_or(nullptr));

    w = {1024, kCPU};
    REQUIRE(w.raw_data());
    REQUIRE(std::as_const(w).raw_data());
}

TEST_CASE("test buffer view", "[buffer]")
{
    using core::Buffer;

    std::vector<int64_t> v{0, 1, 2, 3, 4, 5, 6, 7};

    Buffer b(v.data(), v.size(), kCPU);

    auto c = b.slice(2, 4);
    REQUIRE(c.size() == 4);
    REQUIRE(c.raw_data() == b.data<int64_t>() + 2);

    std::cout << c << std::endl;

    auto d = c.view<int>();

    REQUIRE(d.size() == c.size() * 2);
    REQUIRE(d.raw_data() == c.raw_data());
}

TEST_CASE("test layout", "[layout]")
{
    using core::Layout;

    Layout a;  // default ctor
    REQUIRE(a.size() == 0);
    REQUIRE(a.cosize() == 0);

    Layout b({20, 50});
    REQUIRE(b.size() == 1000);
    REQUIRE(b.cosize() == b.size());
    REQUIRE(to_string(b) == "(20,50):(50,1)");

    Layout c = b.coalesce();
    REQUIRE(c.size() == b.size());
    REQUIRE(c.cosize() == b.cosize());
    REQUIRE(to_string(c) == "(1000):(1)");

    Layout v = b.view({50, 20});
    REQUIRE(v.size() == b.size());
    REQUIRE(v.cosize() == b.cosize());
    REQUIRE(to_string(v) == "(50,20):(20,1)");

    v = b.view({25, -1});
    REQUIRE(to_string(v) == "(25,40):(40,1)");

    v = b.view({5, -1, 5});
    REQUIRE(to_string(v) == "(5,40,5):(200,5,1)");

    v = b.view({-1, 20, 10, 1});
    REQUIRE(to_string(v) == "(5,20,10,1):(200,10,1,1)");

    REQUIRE(to_string(v.coalesce()) == "(1000):(1)");

    auto [s, offset] = b.slice({10, 20}, {-1, -1});
    REQUIRE(to_string(s) == "(10,30):(50,1)");
    REQUIRE(offset == 520);

    v = s.view({2, -1, 3, 10});
    std::cout << v << std::endl;

    std::cout << v.coalesce() << std::endl;

    // v = s.view({30, 10});
    // std::cout << v << std::endl;
}

TEST_CASE("test tensor", "[tensor]")
{
    using core::Tensor;
    using core::Tensor_;
    using core::Allocator;

    Tensor a;
    REQUIRE(!a);

    Tensor_<float> b{{10, 20}, kCPU};
    Tensor_<float> c = b.slice(0, 5);

    std::cout << b << std::endl;

    REQUIRE(c.shape() == std::vector<ssize_t>{5, 20});
    REQUIRE(c.data() == b.data());

    auto d = b.view({2, -1, 10});
    REQUIRE(d.shape() == std::vector<ssize_t>{2, 10, 10});

    // this is typed
    Tensor_<float> x = Tensor_<float>{};
    // while being empty
    REQUIRE(!x);

    if (0) {
        // empty Tensor has invalid type
        Tensor_<float> x = Tensor{};
    }
    a = {};
    x = {};

    Tensor y = core::Buffer{100, kInt32, kCPU};
    REQUIRE(y.ndim() == 1);
    REQUIRE(y.shape(0) == 100);
}

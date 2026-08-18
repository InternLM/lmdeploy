// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/data_format.h"

#include "catch2/catch_test_macros.hpp"

using namespace turbomind;

TEST_CASE("DataFormat default is not quantized", "[data_format]")
{
    DataFormat fmt;
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.rank() == 0);
    REQUIRE(!fmt.scales.present());
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat trivial is not quantized", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kHalf, 1, 1);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.rank() == 2);
    REQUIRE(fmt.block_sizes == std::vector<int>{1, 1});
    REQUIRE(!fmt.scales.present());
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP8 blocked", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kFloat8_e4m3, 128, 128);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat8_e4m3);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 128});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kFloat);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat FP4", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kFloat4_e2m1, 128, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kFloat4_e2m1);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 1});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kUint8);
    REQUIRE(!fmt.zeros.present());
}

TEST_CASE("DataFormat AWQ uint4", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kHalf, kUint4, 128, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.dtype == kUint4);
    REQUIRE(fmt.block_sizes == std::vector<int>{128, 1});
    REQUIRE(fmt.scales.present());
    REQUIRE(fmt.scales.dtype == kHalf);
    REQUIRE(fmt.zeros.present());
    REQUIRE(fmt.zeros.dtype == kHalf);
}

TEST_CASE("DataFormat uint8 quantized", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kBfloat16, kUint8, 64, 1);
    REQUIRE(fmt.is_quantized());
    REQUIRE(fmt.block_sizes == std::vector<int>{64, 1});
    REQUIRE(fmt.scales.dtype == kBfloat16);
    REQUIRE(fmt.zeros.dtype == kBfloat16);
}

TEST_CASE("DataFormat trivial BF16", "[data_format]")
{
    DataFormat fmt = ResolveLinearWeightFormat(kBfloat16, kBfloat16, 1, 1);
    REQUIRE(!fmt.is_quantized());
    REQUIRE(fmt.dtype == kBfloat16);
}

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/dispatch_cache.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/types.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>

static inline bool operator==(const int3& a, const int3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static inline bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x && a.y == b.y;
}

namespace turbomind::gemm {

static inline decltype(auto) as_tuple(const KernelDesc& d)
{
    return std::tie(d.arch,
                    d.op_class,
                    d.type_a,
                    d.type_b,
                    d.type_c,
                    d.order_a,
                    d.order_b,
                    d.order_c,
                    d.striding_a,
                    d.striding_b,
                    d.striding_c,
                    d.pack_a,
                    d.pack_b,
                    d.pack_u,
                    d.pack_v,
                    d.quant_a,
                    d.quant_b,
                    d.policy_a,
                    d.policy_b,
                    d.cta_tile,
                    d.mma_tile,
                    d.cluster_shape,
                    d.align,
                    d.c_tile,
                    d.stages,
                    d.split_k,
                    d.backend,
                    d.transpose,
                    d.group_axis);
}

static inline bool operator==(const QuantDesc& a, const QuantDesc& b)
{
    return a.type == b.type && a.group_size == b.group_size;
}

static inline bool operator==(const KernelDesc& a, const KernelDesc& b)
{
    return as_tuple(a) == as_tuple(b);
}

namespace {

struct Record {
    GemmDesc   gemm;
    KernelDesc kernel;

    int swizzle;
    int splits;
};

}  // namespace

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries)
{

    for (const auto& [g, spec] : entries) {
        Record record{};
        record.gemm    = g;
        record.kernel  = spec.kernel->desc();
        record.splits  = spec.splits;
        record.swizzle = spec.swizzle;

        os.write((const char*)&record, sizeof(record));
    }
}

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<Kernel*>&                   kernels)
{
    is.seekg(0, is.end);
    const auto size_in_bytes = is.tellg();
    is.seekg(0, is.beg);

    if (size_in_bytes % sizeof(Record)) {
        std::cerr << "File size is not a multiple of record size, faild to import records.\n";
    }

    const int n = size_in_bytes / sizeof(Record);

    for (int i = 0; i < n; ++i) {
        Record record;
        is.read((char*)&record, sizeof(Record));

        LaunchSpec spec{};
        spec.splits  = record.splits;
        spec.swizzle = record.swizzle;

        for (const auto& p : kernels) {
            if (p->desc() == record.kernel) {
                spec.kernel = p;
                break;
            }
        }
        if (spec.kernel) {
            entries.emplace_back(record.gemm, spec);
        }
        else {
            std::cerr << "No kernel found for entry " << i << "\n";
        }
    }
}

namespace {

inline decltype(auto) as_tuple(const GemmDesc& d)
{
    return std::tie(d.arch,
                    d.type_a,
                    d.type_b,
                    d.type_c,
                    d.order_a,
                    d.order_b,
                    d.order_c,
                    d.striding_a,
                    d.striding_b,
                    d.striding_c,
                    d.pack_a,
                    d.pack_b,
                    d.pack_u,
                    d.pack_v,
                    d.quant_a.type,
                    d.quant_a.group_size,
                    d.quant_b.type,
                    d.quant_b.group_size,
                    d.batch_dim,
                    d.group_axis,
                    d.m,
                    d.n,
                    d.k,
                    d.num);
    // Note: `d.epilogue` is not used yet
}

}  // namespace

inline bool operator<(const GemmDesc& a, const GemmDesc& b)
{
    return as_tuple(a) < as_tuple(b);
}

int extract_batch_size(GemmDesc& desc)
{
    return std::exchange(desc.batch_dim == 0 ? desc.m : desc.n, 0);
}

void set_batch_size(GemmDesc& desc, int batch_size)
{
    (desc.batch_dim == 0 ? desc.m : desc.n) = batch_size;
}

struct DispatchCache::Impl {

    struct Flat {
        std::vector<std::pair<int, int>> idxs;
        std::vector<LaunchSpec>          specs;
    };

    const std::vector<Kernel*> kernels_;
    std::map<GemmDesc, Flat>   cache_;

    Impl(std::vector<Kernel*> kernels): kernels_(std::move(kernels)) {}

    std::optional<LaunchSpec> Find(GemmDesc desc, bool exact) const
    {
        const int batch_size = extract_batch_size(desc);
        // std::cerr << batch_size << " " << desc.m << " " << desc.n << " " << desc.k << " " << std::boolalpha << exact
        //           << "\n";
        const auto it = cache_.find(desc);
        if (it != cache_.end()) {
            const auto& [idxs, specs] = it->second;
            // Find index via key
            const auto p =
                std::lower_bound(idxs.begin(), idxs.end(), std::make_pair(batch_size, 0), [](auto& a, auto& b) {  //
                    return a.first < b.first;
                });
            // std::cout << it->second.specs.size() << std::endl;
            if (p != idxs.end() && (!exact || p->first == batch_size)) {
                // std::cerr << p->first << " " << p->second << "\n";
                return specs[p->second];
            }
        }
        return {};
    }

    bool Insert(GemmDesc desc, const LaunchSpec& spec)
    {
        const int batch_size = extract_batch_size(desc);

        auto it = cache_.find(desc);
        if (it == cache_.end()) {
            it = cache_.emplace_hint(it, desc, Flat{});
        }
        auto& [idxs, specs] = it->second;
        // Find index via key
        const auto p =
            std::lower_bound(idxs.begin(), idxs.end(), std::make_pair(batch_size, 0), [](auto& a, auto& b) {  //
                return a.first < b.first;
            });
        // Exact match, skip
        if (p != idxs.end() && p->first == batch_size) {
            return false;
        }
        // Insert
        idxs.insert(p, {batch_size, (int)specs.size()});
        specs.push_back(spec);
        return true;
    }

    int Export(std::ostream& os) const
    {
        std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
        for (const auto& [desc, flat] : cache_) {
            auto tmp = desc;
            for (const auto& [batch_size, index] : flat.idxs) {
                set_batch_size(tmp, batch_size);
                entries.emplace_back(tmp, flat.specs[index]);
            }
        }
        Summary(entries);
        ExportDispatchCache(os, entries);
        return entries.size();
    }

    int Import(std::istream& is)
    {
        std::vector<std::pair<GemmDesc, LaunchSpec>> entries;
        ImportDispatchCache(is, entries, kernels_);
        Summary(entries);
        for (auto [desc, spec] : entries) {
            const int batch_size = extract_batch_size(desc);
            auto      it         = cache_.find(desc);
            if (it == cache_.end()) {
                it = cache_.emplace_hint(it, desc, Flat{});
            }
            auto& [idxs, specs] = it->second;
            // Order is not maintained at this point
            idxs.emplace_back(batch_size, (int)specs.size());
            specs.push_back(spec);
        }
        // Sort indices and deduplicate
        for (auto& [desc, flat] : cache_) {
            auto& [idxs, specs] = flat;
            std::stable_sort(idxs.begin(), idxs.end(), [](auto a, auto b) { return a.first < b.first; });
            idxs.erase(std::unique(idxs.begin(), idxs.end(), [](auto a, auto b) { return a.first == b.first; }),
                       idxs.end());
            // Remove unreferenced specs and update spec indices
            std::vector<LaunchSpec> tmp;
            for (auto& [key, val] : idxs) {
                int old = std::exchange(val, tmp.size());
                tmp.push_back(specs[old]);
            }
            specs = std::move(tmp);
        }
        return entries.size();
    }

    // Print a summary of how many cases a kernel is used
    void Summary(const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries) const
    {
        std::vector<Kernel*> uses{nullptr};
        std::copy(kernels_.begin(), kernels_.end(), std::back_inserter(uses));

        for (const auto& [_, s] : entries) {
            uses.push_back(s.kernel);
        }
        std::sort(uses.begin(), uses.end());
        std::vector<std::pair<int, Kernel*>> count;
        for (size_t i = 1; i < uses.size(); ++i) {
            if (uses[i] != uses[i - 1]) {
                count.emplace_back(-1, uses[i]);
            }
            ++count.back().first;
        }
        std::sort(count.begin(), count.end(), std::greater<>{});
        for (const auto& [n, k] : count) {
            std::cout << k->name() << ": " << n << "\n";
        }
    }
};

DispatchCache::DispatchCache(std::vector<Kernel*> kernels): impl_(std::make_unique<Impl>(std::move(kernels))) {}

DispatchCache::~DispatchCache() = default;

std::optional<LaunchSpec> DispatchCache::Find(const GemmDesc& desc) const
{
    return impl_->Find(desc, true);
}

std::optional<LaunchSpec> DispatchCache::LowerBound(const GemmDesc& desc) const
{
    return impl_->Find(desc, false);
}

bool DispatchCache::Insert(const GemmDesc& desc, const LaunchSpec& spec)
{
    return impl_->Insert(desc, spec);
}

int DispatchCache::Export(std::ostream& os) const
{
    return impl_->Export(os);
}

int DispatchCache::Import(std::istream& is)
{
    return impl_->Import(is);
}

}  // namespace turbomind::gemm

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
                    d.align,
                    d.c_tile,
                    d.stages,
                    d.split_k);
}

static inline bool operator==(const QuantDesc& a, const QuantDesc& b)
{
    return a.type == b.type && a.group_size == b.group_size;
}

static inline bool operator==(const KernelDesc& a, const KernelDesc& b)
{
    return as_tuple(a) == as_tuple(b);
}

template<class... Ts>
static inline void export_impl(std::ostream& os, const Ts&... ts)
{
    ((os << static_cast<int>(ts) << " "), ...);
}

template<class T>
static inline void import_value(std::istream& is, T& value)
{
    int token{};
    is >> token;
    value = static_cast<T>(token);
}

template<class... Ts>
static inline void import_impl(std::istream& is, Ts&... ts)
{
    (import_value(is, ts), ...);
}

void ExportDispatchCache(std::ostream& os, const std::vector<std::pair<GemmDesc, LaunchSpec>>& entries)
{
    for (const auto& [g, spec] : entries) {
        // GEMM desc
        export_impl(os,
                    g.arch,
                    g.type_a,
                    g.type_b,
                    g.type_c,
                    g.order_a,
                    g.order_b,
                    g.order_c,
                    g.pack_a,
                    g.pack_b,
                    g.pack_u,
                    g.pack_v,
                    g.quant_a.type,
                    g.quant_a.group_size,
                    g.quant_b.type,
                    g.quant_b.group_size,
                    g.epilogue,
                    g.m,
                    g.n,
                    g.k,
                    g.batch_dim);
        // Kernel desc
        auto& k = spec.kernel->desc();
        export_impl(os,
                    k.arch,
                    k.op_class,
                    k.cta_tile.x,
                    k.cta_tile.y,
                    k.cta_tile.z,
                    k.mma_tile.x,
                    k.mma_tile.y,
                    k.mma_tile.z,
                    k.stages,
                    k.align.x,
                    k.align.y,
                    k.align.z,
                    k.policy_a,
                    k.policy_b,
                    k.c_tile.x,
                    k.c_tile.y,
                    k.split_k);
        // Runtime params
        export_impl(os, spec.swizzle, spec.splits);
        os << std::endl;
    }
}

void ImportDispatchCache(std::istream&                                 is,
                         std::vector<std::pair<GemmDesc, LaunchSpec>>& entries,
                         const std::vector<Kernel*>&                   kernels)
{
    std::string line;
    while (std::getline(is, line)) {
        std::cout << line << std::endl;
        std::stringstream ss(line);
        GemmDesc          g{};
        import_impl(ss,
                    g.arch,
                    g.type_a,
                    g.type_b,
                    g.type_c,
                    g.order_a,
                    g.order_b,
                    g.order_c,
                    g.pack_a,
                    g.pack_b,
                    g.pack_u,
                    g.pack_v,
                    g.quant_a.type,
                    g.quant_a.group_size,
                    g.quant_b.type,
                    g.quant_b.group_size,
                    g.epilogue,
                    g.m,
                    g.n,
                    g.k,
                    g.batch_dim);
        KernelDesc k{};
        k.type_a  = g.type_a;
        k.type_b  = g.type_b;
        k.type_c  = g.type_c;
        k.pack_a  = g.pack_a;
        k.pack_b  = g.pack_b;
        k.pack_u  = g.pack_u;
        k.pack_v  = g.pack_v;
        k.order_a = g.order_a;
        k.order_b = g.order_b;
        k.order_c = g.order_c;
        k.quant_a = g.quant_a;
        k.quant_b = g.quant_b;
        import_impl(ss,
                    k.arch,
                    k.op_class,
                    k.cta_tile.x,
                    k.cta_tile.y,
                    k.cta_tile.z,
                    k.mma_tile.x,
                    k.mma_tile.y,
                    k.mma_tile.z,
                    k.stages,
                    k.align.x,
                    k.align.y,
                    k.align.z,
                    k.policy_a,
                    k.policy_b,
                    k.c_tile.x,
                    k.c_tile.y,
                    k.split_k);
        LaunchSpec spec{};
        import_impl(ss, spec.swizzle, spec.splits);
        for (const auto& p : kernels) {
            if (p->desc() == k) {
                spec.kernel = p;
                break;
            }
        }
        if (spec.kernel) {
            entries.emplace_back(g, spec);
        }
        else {
            std::cerr << "No kernel found for entry: " << line << "\n";
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
                    d.pack_a,
                    d.pack_b,
                    d.pack_u,
                    d.pack_v,
                    d.quant_a.type,
                    d.quant_a.group_size,
                    d.quant_b.type,
                    d.quant_b.group_size,
                    d.m,
                    d.n,
                    d.k,
                    d.batch_dim);
    // d.epilogue
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
        // std::cerr << batch_size << " " << desc.m << " " << desc.n << " " << desc.k << "\n";
        const auto it = cache_.find(desc);
        if (it != cache_.end()) {
            const auto& [idxs, specs] = it->second;
            // Find index via key
            const auto p =
                std::lower_bound(idxs.begin(), idxs.end(), std::make_pair(batch_size, 0), [](auto& a, auto& b) {  //
                    return a.first < b.first;
                });
            // std::cerr << p->first << " " << p->second << "\n";
            if (p != idxs.end() && (!exact || p->first == batch_size)) {
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
            const auto cmp      = [](auto& a, auto& b) {  //
                return a.first < b.first;
            };
            std::stable_sort(idxs.begin(), idxs.end(), cmp);
            idxs.erase(std::unique(idxs.begin(), idxs.end(), cmp), idxs.end());
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

// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/tune/sampler.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace turbomind::gemm {

template<class Op>
inline static bool cmp(const int3& a, const int3& b, Op op)
{
    return op(std::tie(a.x, a.y, a.z), std::tie(b.x, b.y, b.z));
}

std::vector<LaunchSpec> Sampler::Run(std::vector<LaunchSpec> specs, const Launcher& launcher, cudaStream_t stream)
{
    std::vector<std::vector<LaunchSpec*>> clusters;  // ptr into `specs`
    if (k_clusters_) {
        clusters = Cluster(specs);
    }
    else {
        for (auto& s : specs) {
            clusters.push_back({&s});
        }
    }
    // std::cout << "k_clusters=" << k_clusters_ << ", #specs" << specs.size() << ", #clusters" << clusters.size() <<
    // "\n";

    std::vector<LaunchSpec> s_1;
    for (const auto& c : clusters) {
        s_1.push_back(*c.front());
    }

    auto m_1 = measurer_.Measure(s_1, launcher, stream);

    auto idxs = ArgSort(m_1);

    if (k_clusters_) {
        const auto top_k = std::min(k_clusters_, (int)idxs.size());
        idxs.resize(top_k);

        std::vector<LaunchSpec> s_2;
        for (const auto& idx : idxs) {
            auto& cluster = clusters[idx];
            // Skip cluster leader
            for (size_t j = 1; j < cluster.size(); ++j) {
                s_2.push_back(*cluster[j]);
            }
        }

        // std::cout << "#s_2=" << s_2.size() << "\n";

        auto m_2 = measurer_.Measure(s_2, launcher, stream);
        // Merge measurements of the 2 runs
        m_2.insert(m_2.end(), m_1.begin(), m_1.end());
        s_2.insert(s_2.end(), s_1.begin(), s_1.end());
        m_1.swap(m_2);
        s_1.swap(s_2);
    }

    idxs = ArgSort(m_1);
    std::vector<LaunchSpec> ret;
    for (const auto& i : idxs) {
        s_1[i].measured = m_1[i].mean;
        ret.push_back(s_1[i]);
    }

    return ret;
}

std::vector<int> Sampler::ArgSort(const std::vector<Measurement>& ms)
{
    std::vector<int> idxs(ms.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {  //
        return ms[i].mean < ms[j].mean;
    });
    return idxs;
}

std::vector<std::vector<LaunchSpec*>> Sampler::Cluster(std::vector<LaunchSpec>& specs)
{
    std::vector<LaunchSpec*> ptrs;
    for (auto& s : specs) {
        ptrs.push_back(&s);
    }

    auto less = [](LaunchSpec* u, LaunchSpec* v) {
        const auto& a = u->kernel->desc();
        const auto& b = v->kernel->desc();
        if (!cmp(a.cta_tile, b.cta_tile, std::equal_to<>{})) {
            return cmp(a.cta_tile, b.cta_tile, std::less<>{});
        }
        if (!cmp(a.mma_tile, b.mma_tile, std::equal_to<>{})) {
            return cmp(a.mma_tile, b.mma_tile, std::less<>{});
        }
        const auto pa = std::tie(a.policy_a, a.policy_b);
        const auto pb = std::tie(b.policy_a, b.policy_b);
        if (pa != pb) {
            return pa < pb;
        }
        return u->splits < v->splits;
    };

    auto equal = [&](LaunchSpec* u, LaunchSpec* v) {  //
        return !less(u, v) && !less(v, u);
    };

    std::stable_sort(ptrs.begin(), ptrs.end(), less);

    std::vector<std::vector<LaunchSpec*>> clusters{{ptrs[0]}};

    int p = 0;
    for (size_t i = 1; i < ptrs.size(); ++i) {
        if (equal(ptrs[p], ptrs[i])) {
            clusters.back().push_back(ptrs[i]);
        }
        else {
            clusters.push_back({ptrs[i]});
            p = i;
        }
    }

    return clusters;
}

}  // namespace turbomind::gemm
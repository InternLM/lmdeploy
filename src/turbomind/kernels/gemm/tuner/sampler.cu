// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/desc.h"
#include "src/turbomind/kernels/gemm/kernel.h"
#include "src/turbomind/kernels/gemm/tuner/sampler.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace turbomind::gemm {

template<class Cmp>
static std::vector<int> ArgSort(size_t size, const Cmp& cmp)
{
    std::vector<int> idxs(size);
    std::iota(idxs.begin(), idxs.end(), 0);
    std::stable_sort(idxs.begin(), idxs.end(), cmp);
    return idxs;
}

std::vector<LaunchSpec> Sampler::Run(std::vector<LaunchSpec> specs, const Launcher& launcher, cudaStream_t stream)
{
    std::vector<std::vector<LaunchSpec>> clusters;  // ptr into `specs`
    if (k_clusters_) {
        clusters = Cluster(specs, ClusteringParam{true, true});
    }
    else {
        for (auto& s : specs) {
            clusters.push_back({s});
        }
    }
    // std::cout << "k_clusters=" << k_clusters_ << ", #specs" << specs.size() << ", #clusters" << clusters.size() <<
    // "\n";

    std::vector<LaunchSpec> s_1;
    for (const auto& c : clusters) {
        s_1.push_back(c.front());
    }

    auto m_1 = measurer_.Measure(s_1, launcher, stream);

    auto idxs = ArgSort(m_1.size(), [&](int i, int j) { return m_1[i].mean < m_1[j].mean; });

    if (k_clusters_) {
        const auto top_k = std::min(k_clusters_, (int)idxs.size());
        idxs.resize(top_k);

        std::vector<LaunchSpec> s_2;
        for (const auto& idx : idxs) {
            auto& cluster = clusters[idx];
            // Skip cluster leader
            for (size_t j = 1; j < cluster.size(); ++j) {
                s_2.push_back(cluster[j]);
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

    idxs = ArgSort(m_1.size(), [&](int i, int j) { return m_1[i].mean < m_1[j].mean; });

    std::vector<LaunchSpec> ret;
    for (const auto& i : idxs) {
        s_1[i].measured = m_1[i].mean;
        ret.push_back(s_1[i]);
    }

    return ret;
}

}  // namespace turbomind::gemm

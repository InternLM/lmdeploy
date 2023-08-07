// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

struct Metric {
    bool feasible;

    int   max_active_ctas;
    int   active_ctas;
    float occupancy;
    float waves;
    int   stages;

    float m_iter;
    float n_iter;
    float tile_efficiency;
    float wave_efficiency;
    float nice;
    float cost;
    float normalized;

    std::array<int, 3> cta_shape;
    std::array<int, 3> warp_shape;
    int                warps;

    float time;
    int   count;

    int id;

    int best;
};

inline void DumpMetrics(std::ostream& os, const std::vector<Metric>& metrics, const std::vector<int>& indices)
{
    const std::vector<int>         widths{12, 14, 8, 8, 10, 12, 12, 12, 12, 10, 12, 12, 15, 12, 12, 12, 12, 12, 8};
    const std::vector<std::string> names{"cta shape",
                                         "warp shape",
                                         "warps",
                                         "id",
                                         "feasible",
                                         "max_ctas",
                                         "active_ctas",
                                         "occupancy",
                                         "waves",
                                         "stages",
                                         "m_iter",
                                         "n_iter",
                                         "%tile",
                                         "%wave",
                                         "nice",
                                         "cost",
                                         "normalized",
                                         "time",
                                         "best"};
    for (size_t i = 0; i < names.size(); ++i) {
        os << std::setw(widths[i]) << names[i];
    }
    os << "\n";

    for (size_t i = 0; i < metrics.size(); ++i) {
        auto& metric = indices.empty() ? metrics[i] : metrics[indices[i]];
        int   c      = 0;

        {
            std::stringstream ss;
            ss << std::setw(4) << metric.cta_shape[0] << std::setw(4) << metric.cta_shape[1] << std::setw(4)
               << metric.cta_shape[2];
            os << std::setw(widths[c++]) << ss.str();
        }

        {
            std::stringstream ss;
            ss << std::setw(4) << metric.warp_shape[0] << std::setw(4) << metric.warp_shape[1] << std::setw(4)
               << metric.warp_shape[2];
            os << std::setw(widths[c++]) << ss.str();
        }

        os << std::setw(widths[c++]) << metric.warps;

        os << std::setw(widths[c++]) << metric.id;
        os << std::setw(widths[c++]) << metric.feasible;
        os << std::setw(widths[c++]) << metric.max_active_ctas;
        os << std::setw(widths[c++]) << metric.active_ctas;
        os << std::setw(widths[c++]) << metric.occupancy;
        os << std::setw(widths[c++]) << metric.waves;
        os << std::setw(widths[c++]) << metric.stages;
        os << std::setw(widths[c++]) << metric.m_iter;
        os << std::setw(widths[c++]) << metric.n_iter;
        os << std::setw(widths[c++]) << metric.tile_efficiency;
        os << std::setw(widths[c++]) << metric.wave_efficiency;
        os << std::setw(widths[c++]) << metric.nice;
        os << std::setw(widths[c++]) << metric.cost;
        os << std::setw(widths[c++]) << metric.normalized;
        if (metric.count) {
            os << std::setw(widths[c]) << metric.time * 1000 / metric.count;
        }
        c++;
        if (metric.best) {
            os << std::setw(widths[c]) << '*';
        }
        c++;
        os << "\n";
    }
}

}  // namespace turbomind
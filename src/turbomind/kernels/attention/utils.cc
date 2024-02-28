

#include "utils.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <tuple>

namespace turbomind {

int GetSplitCount(
    int max_split_cnt, int grid_size, int max_active_ctas, int sm_count, int max_wave_cnt, float alpha, float beta)
{

    const float scale = (float)grid_size / (sm_count * max_active_ctas);

    auto eval = [&](int s) -> std::tuple<float, float, int> {
        float waves = std::ceil(scale * s);
        float cost  = std::numeric_limits<float>::infinity();
        if (s == 1 || waves <= max_wave_cnt) {
            cost = (alpha / s + beta) * waves;
        }
        return {cost, scale * s, s};
    };

    std::tuple<float, float, int> best{std::numeric_limits<float>::infinity(), 0.f, 0};

    auto print = [](auto& x) {  //
        // printf("%d %f %f\n", std::get<2>(x), std::get<1>(x), std::get<0>(x));
    };

    for (int i = 1; i <= max_split_cnt; ++i) {
        auto res = eval(i);
        if (std::isinf(std::get<0>(res))) {
            break;
        }
        print(res);
        if (res < best) {
            best = res;
        }
    }

    print(best);

    return std::get<int>(best);
}

}  // namespace turbomind

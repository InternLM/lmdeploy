#pragma once

namespace turbomind {

int GetSplitCount(int   max_split_cnt,
                  int   grid_size,
                  int   max_active_ctas,
                  int   sm_count,
                  int   max_wave_cnt,
                  float alpha = 1,
                  float beta  = 1e-3);

}
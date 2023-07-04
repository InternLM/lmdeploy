# PTQ 量化测试结果

测试对象为内部早期的 100B 模型。尽管模型暂不开放，测试数据仍能展示量化方法对此模型的影响。
测试方法：
1. 运行 `deploy.py`，切分 100B 模型到 8 个 GPU 上
2. 运行量化脚本，得到量化参数，放到 weights 目录
3. 修改配置文件，使 [kCacheKVInt8](../src/turbomind/models/llama/llama_utils.h) 选项生效
4. 执行测试数据集，和 fp16 版本对比精度和显存使用情况

## 显存降低

随着 batch_size 增加，`kCacheKVInt8` 可以节约更多显存，从而降低部署成本。

| batch | int8 memory(GB/GPU) | fp16 memory(GB/GPU) |
| :-: | :-: | :-: |
| 16 | 40 | 43 |
| 32 | 48 | 60 |


## 精度影响

以下是 `kCacheKVInt8` 方法仅用 c4 数据集量化，在其他数据集的精度损失，数值仅供参考。

| task | dataset | version | metric | diff |
| :-: | :-: | :-: | :-: | :-: |
| Exam             | ceval           | -         | avg_accuracy | -0.43 |
| Exam             | ceval-hard      | -         | avg_accuracy | 2.24 |
| ChineseUniversal | CMRC_dev        | v1-65aa5c | score        | -2.99 |
| ChineseUniversal | DRCD_dev        | v1-65aa5c | score        | -1.14 |
| ChineseUniversal | afqmc-dev       | v1-bbbabc | accuracy     | 1.67 |
| ChineseUniversal | bustm-dev       | v1-ecded6 | accuracy     | 10.62 |
| ChineseUniversal | bustm-test      | v1-ecded6 | accuracy     | 14.90 |
| ChineseUniversal | chid-dev        | v1-ffc5eb | accuracy     | -5.94 |
| ChineseUniversal | chid-test       | v1-ffc5eb | accuracy     | -4.19 |
| ChineseUniversal | cluewsc-dev     | v1-b88a63 | accuracy     | -4.40 |
| ChineseUniversal | cluewsc-test    | v1-b88a63 | accuracy     | -2.56 |
| ChineseUniversal | eprstmt-dev     | v1-99cf6f | accuracy     | 1.87 |
| ChineseUniversal | eprstmt-test    | v1-99cf6f | accuracy     | 1.48 |
| Completion       | lambada         | v1-678ebd | accuracy     | -1.65 |
| Completion       | story_cloze     | v1-f92a41 | accuracy     | -0.11 |
| EnglishUniversal | AX_b            | v1-78e4c2 | accuracy     | -1.27 |
| EnglishUniversal | AX_g            | v1-ccfc17 | accuracy     | -2.81 |
| EnglishUniversal | BoolQ           | v1-2c7cf3 | accuracy     | -4.22 |
| EnglishUniversal | CB              | v1-f60fbb | accuracy     | 0.00 |
| EnglishUniversal | COPA            | v1-d3a03c | accuracy     | -2.00 |
| EnglishUniversal | MultiRC         | v1-560d31 | accuracy     | -8.79 |
| EnglishUniversal | ReCoRD          | v1-5a2219 | score        | -2.09 |
| EnglishUniversal | RTE             | v1-ccfc17 | accuracy     | -3.25 |
| EnglishUniversal | WiC             | v1-019721 | accuracy     | -6.74 |
| EnglishUniversal | WSC             | v1-57571c | accuracy     | -5.77 |
| EnglishUniversal | race-middle     | v1-0c5c3c | accuracy     | -1.19 |
| EnglishUniversal | race-high       | v1-0c5c3c | accuracy     | -1.06 |
| Reasoning        | gsm8k_main      | v1-3d5be1 | accuracy     | -8.80 |
| QA               | hellaswag       | v1-3e134d | accuracy     | -1.45 |
| QA               | piqa            | v1-362133 | accuracy     | -1.53 |
| QA               | winogrande      | v1-a2f53f | accuracy     | -0.79 |
| QA               | openbookqa      | v1-8587d7 | accuracy     | -7.00 |
| QA               | openbookqa_fact | v1-4e92f0 | accuracy     | -14.00 |
| QA               | nq              | v1-d2370e | score        | -2.16 |
| QA               | triviaqa        | v1-ead882 | score        | -0.43 |
| Security         | crows_pairs     | v1-8fe12f | accuracy     | 11.08 |
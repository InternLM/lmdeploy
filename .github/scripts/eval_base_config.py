from copy import deepcopy

from mmengine.config import read_base

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import \
        bbh_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.ceval.ceval_ppl import \
        ceval_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import \
        cmmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.crowspairs.crowspairs_ppl import \
        crowspairs_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.drop.drop_gen_a2697c import \
        drop_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gpqa.gpqa_ppl_6bf57a import \
        gpqa_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import \
        hellaswag_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import \
        mmlu_datasets  # noqa: F401, E501
    from opencompass.configs.datasets.race.race_ppl_a138cd import \
        race_datasets  # noqa: F401, E501
    # read models
    from opencompass.configs.models.baichuan.hf_baichuan_7b import \
        models as hf_baichuan_7b  # noqa: F401, E501
    from opencompass.configs.models.gemma.hf_gemma_7b import \
        models as hf_gemma_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_5_7b import \
        models as hf_internlm2_5_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_7b import \
        models as hf_internlm2_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm2_20b import \
        models as hf_internlm2_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_7b import \
        models as hf_internlm_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.hf_internlm_20b import \
        models as hf_internlm_20b  # noqa: F401, E501
    from opencompass.configs.models.hf_internlm.lmdeploy_internlm2_5_7b import \
        models as lmdeploy_internlm2_5_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama2_7b import \
        models as hf_llama2_7b  # noqa: F401, E501
    from opencompass.configs.models.hf_llama.hf_llama3_8b import \
        models as hf_llama3_8b  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mistral_7b_v0_1 import \
        models as hf_mistral_7b_v0_1  # noqa: F401, E501
    from opencompass.configs.models.mistral.hf_mixtral_8x7b_v0_1 import \
        models as hf_mixtral_8x7b_v0_1  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen1_5_7b import \
        models as hf_qwen1_5_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen2_7b import \
        models as hf_qwen2_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.hf_qwen_7b import \
        models as hf_qwen_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen1_5_7b import \
        models as lmdeploy_qwen1_5_7b  # noqa: F401, E501
    from opencompass.configs.models.qwen.lmdeploy_qwen2_7b import \
        models as lmdeploy_qwen2_7b  # noqa: F401, E501
    # and output the results in a chosen format
    from opencompass.configs.summarizers.medium import \
        summarizer  # noqa: F401, E501

turbomind_qwen1_5_7b = deepcopy(*lmdeploy_qwen1_5_7b)
turbomind_qwen2_7b = deepcopy(*lmdeploy_qwen2_7b)
turbomind_internlm2_5_7b = deepcopy(*lmdeploy_internlm2_5_7b)

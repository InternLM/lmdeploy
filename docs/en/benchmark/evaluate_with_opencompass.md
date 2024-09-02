# Evaluate LLMs with OpenCompass

The LLMs accelerated by lmdeploy can be evaluated with OpenCompass.

## Setup

In this part, we are going to setup the environment for evaluation.

### Install lmdeploy

Please follow the [installation guide](../get_started/installation.md) to install lmdeploy.

### Install OpenCompass

Install OpenCompass from source. Refer to [installation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) for more information.

```shell
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

At present, you can check the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html#)
to get to know the basic usage of OpenCompass.

### Download datasets

Download the core datasets

```shell
# Run in the OpenCompass directory
cd opencompass
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

## Prepare Evaluation Config

OpenCompass uses the configuration files as the OpenMMLab style. One can define a python config and start evaluating at ease.
OpenCompass has supported the evaluation for lmdeploy's TurboMind engine using python API.

### Dataset Config

In the home directory of OpenCompass, we are writing the config file `$OPENCOMPASS_DIR/configs/eval_lmdeploy.py`.
We select multiple predefined datasets and import them from OpenCompass base dataset configs as `datasets`.

```python
from mmengine.config import read_base


with read_base():
    # choose a list of datasets
    from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets
    from .datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets
    # and output the results in a chosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
```

### Model Config

This part shows how to setup model config for LLMs. Let's check some examples:

`````{tabs}
````{tab} internlm-20b

```python
from opencompass.models.turbomind import TurboMindModel

internlm_20b = dict(
        type=TurboMindModel,
        abbr='internlm-20b-turbomind',
        path="internlm/internlm-20b",  # this path should be same as in huggingface
        engine_config=dict(session_len=2048,
                           max_batch_size=8,
                           rope_scaling_factor=1.0),
        gen_config=dict(top_k=1, top_p=0.8,
                        temperature=1.0,
                        max_new_tokens=100),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        concurrency=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

models = [internlm_20b]
```

````

````{tab} internlm-chat-20b

For Chat models, you have to pass `meta_template` for chat models. Different Chat models may have different `meta_template` and it's important
to keep it the same as in training settings. You can read [meta_template](https://opencompass.readthedocs.io/en/latest/prompt/meta_template.html) for more information.


```python
from opencompass.models.turbomind import TurboMindModel

internlm_meta_template = dict(round=[
    dict(role='HUMAN', begin='<|User|>:', end='\n'),
    dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
],
                              eos_token_id=103028)

internlm_chat_20b = dict(
    type=TurboMindModel,
    abbr='internlm-chat-20b-turbomind',
    path='internlm/internlm-chat-20b',
    engine_config=dict(session_len=2048,
                       max_batch_size=8,
                       rope_scaling_factor=1.0),
    gen_config=dict(top_k=1,
                    top_p=0.8,
                    temperature=1.0,
                    max_new_tokens=100),
    max_out_len=100,
    max_seq_len=2048,
    batch_size=8,
    concurrency=8,
    meta_template=internlm_meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str='<eoa>'
)

models = [internlm_chat_20b]

```

````

`````

**Note**

- If you want to pass more arguments for `engine_config`和`gen_config` in the evaluation config file, please refer to [TurbomindEngineConfig](https://github.com/InternLM/lmdeploy/blob/061f99736544c8bf574309d47baf574b69ab7eaf/lmdeploy/messages.py#L114)
  and [EngineGenerationConfig](https://github.com/InternLM/lmdeploy/blob/061f99736544c8bf574309d47baf574b69ab7eaf/lmdeploy/messages.py#L56)

## Execute Evaluation Task

After defining the evaluation config, we can run the following command to start evaluating models.
You can check [Execution Task](https://opencompass.readthedocs.io/en/latest/user_guides/experimentation.html#task-execution-and-monitoring)
for more arguments of `run.py`.

```shell
# in the root directory of opencompass
python3 run.py configs/eval_lmdeploy.py --work-dir ./workdir
```

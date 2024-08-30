# 如何使用OpenCompass测评LLMs

LMDeploy设计了TurboMind推理引擎用来加速大模型推理，其推理精度也支持使用OpenCompass测评。

## 准备

我们将配置用于测评的环境

### 安装 lmdeploy

请参考[安装指南](../get_started/installation.md)安装 lmdeploy

### 安装 OpenCompass

执行如下脚本，从源码安装OpenCompass。更多安装方式请参考[installation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html)。

```shell
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
```

如果想快速了解OpenCompass基本操作，可翻阅[Quick Start](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html#)

### 下载数据集

OpenCompass提供了多个版本的数据集，在这里我们下载如下版本数据集

```shell
# 切换到OpenCompass根目录
cd opencompass
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

## 准备测评配置文件

OpenCompass采用OpenMMLab风格的配置文件来管理模型和数据集，用户只需添加简单的配置就可以快速开始测评。OpenCompass已支持通过python API来
测评TurboMind推理引擎加速的大模型。

### 数据集配置

在OpenCompass根目录，准备测评配置文件`$OPENCOMPASS_DIR/configs/eval_lmdeploy.py`。

在配置文件开始，导入如下OpenCompass支持的数据集`datasets`和格式化输出测评结果的`summarizer`。

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

### 模型配置

这个部分展示如何在测评配置文件中添加模型配置。让我们来看几个示例：

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

对于Chat类大模型，用户需要在配置文件中指定`meta_template`，该设置需要与训练设置对齐，可翻阅[meta_template](https://opencompass.readthedocs.io/en/latest/prompt/meta_template.html) 查看其介绍。

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

**注**

- 如果想在测评配置文件中`engine_config`和`gen_config`字段传递更多参数，请参考[TurbomindEngineConfig](https://github.com/InternLM/lmdeploy/blob/061f99736544c8bf574309d47baf574b69ab7eaf/lmdeploy/messages.py#L114) 和 [EngineGenerationConfig](https://github.com/InternLM/lmdeploy/blob/061f99736544c8bf574309d47baf574b69ab7eaf/lmdeploy/messages.py#L56)

## 执行测评任务

完成测评配置文件编写后，在OpenCompass根目录执行`run.py`脚本，指定工作目录即可开启测评任务。
测评脚本更多参数可参考[执行测评](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html#id1)

```shell
# in the root directory of opencompass
python3 run.py configs/eval_lmdeploy.py --work-dir ./workdir
```

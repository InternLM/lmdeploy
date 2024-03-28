# lmdeploy.vl 新模型支持

目前，有一批 VLM 模型采用如下图所示的架构。图片经过 Vison Encoder 得到图片特征，之后经过 Projection 映射到文本的特征空间。最后，将图片特征与文本特征拼接后送入 LLM 进行推理。这类 VLM 模型有一个特点，即拼接后的特征送入 LLM 推理时并不区分特征的类型，两种特征之间没有交互计算。

![arch](https://llava-vl.github.io/images/llava_arch.png)

对于此类架构的模型，使用 LMDeploy 可以很方便的添加新模型的支持。

## 模型支持

VLM 整体流程如下所示，其中 `ImageEncoder` 负责收集请求中的图片，进行组 batch 操作，`VisionModel` 进行特征的提取。`VLChatTemplateWrapper` 负责将用户输入的 prompt 以及图片转化为 `token_ids` 和 `embedding position`，最后 LLM 利用这些信息进行推理。整体来说，对于新模型的支持，用户需要实现两个部分：

- VisonModel
- VLChatTemplateWrapper

```
+-----------------------+                       +--------+
| ImageEncoder          |                       |        |
|                       |       embedding       |        |
|        +------------+ |---------------------> |        |
|        | VisonModel | |                       |        |
|        +------------+ |                       |        |
+-----------------------+                       |  LLM   |
                                                |        |
+-----------------------+                       |        |
| VLChatTemplateWrapper |                       |        |
|                       |      token_ids        |        |
|       +-------------+ |---------------------> |        |
|       |ChatTemplate | |  embedding position   |        |
|       +-------------+ |                       |        |
+-----------------------+                       +--------+
```

> \[!NOTE\]
>
> 一般 VLM 模型有一个对应的不带图片输入的LLM 模型，如 Qwen-VL-Chat 和 Qwen-7B-Chat，请先确保这个 LLM 模型可以被 TurboMind 引擎推理，或者他的模型结构和 TurboMind 已支持的模型结构相同。
>
> TurboMind 支持的模型可参考[这里](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/turbomind/supported_models.py)

下面以 Qwen/Qwen-VL-Chat 模型为例，展示如何使用 LMDeploy 添加这类模型的支持。

### VisonModel

视觉模型负责图片特征的抽取，得益于 huggingface 的广泛使用，大部分模型都有 transformers 版本。借助 transformers 的 api, 我们可以仅加载模型的 vision 部分，并借助模型本身的函数，对图片进行推理。

添加新的视觉模型，主要需要修改两个地方:

1. 抽取 VLM 模型对应的 vision 模型，并实现 `forward` 特征抽取函数
2. 修改 `load_vl_model` 函数, 使 VLM 模型在加载时可以找到对应的 VisionModel 模型

下面是 Qwen/Qwen-VL-Chat 对应的模型视觉部分。

```python
# lmdeploy/vl/model/qwen.py
class QwenVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        # init an empty model
        with init_empty_weights():
            config = AutoConfig.from_pretrained(self.model_path,
                                                trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            # del unused module, only keep the vision part.
            del model.lm_head
            for key in ['wte', 'h', 'ln_f']:
                setattr(model.transformer, key, None)

        # move model to cpu and load weight
        model.to_empty(device='cpu')
        load_model_from_weight_files(model, self.model_path)

        # move model to gpu
        self.model = model.transformer.visual
        self.model.to(self.device).eval().half()

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.model.image_transform(x) for x in outputs]
        outputs = torch.stack(outputs, dim=0)
        outputs = self.model(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
```

为了在加载 Qwen/Qwen-VL-Chat 模型时可以正确找到 `QwenVisionModel`，需要修改 [`load_vl_model`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/builder.py) 函数，这里我们根据 `architectures` 来判断。

### VLChatTemplateWrapper

VLChatTemplateWrapper 的作用是在 prompt 中加入代表图片的特殊 token，以便在 prompt 转化为 token_ids 的时候得到视觉特征的插入位置。对于新模型的支持，用户需要重写 `append_image_token` 这个函数，实现 VLM 模型对应的逻辑。

添加新的 VLChatTemplateWrapper，主要需要修改两个地方:

1. 根据 VLM 模型的对话模版，将代表图像的特殊 token 插入到 prompt 中
2. 修改 `get_vl_prompt_template`, 使 VLM 模型在加载时可以找到对应的 VLChatTemplateWrapper

下面是 Qwen/Qwen-VL-Chat 对应的 VLChatTemplateWrapper 部分。

```python
class QwenVLChatTemplateWrapper(VLChatTemplateWrapper):
    """Qwen vl chat template."""

    def append_image_token(self, prompt, num_images: int):
        """append image tokens to user prompt."""
        res = ''
        for i in range(num_images):
            res += f'Picture {str(i)}:{IMAGE_TOKEN}\n'
        res = res + prompt
        return res
```

为了在加载 Qwen/Qwen-VL-Chat 模型时可以正确找到 `VLChatTemplateWrapper`，需要修改 [`get_vl_prompt_template`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/templates.py) 函数，这里我们根据 `architectures` 来判断。

> \[!NOTE\]
>
> `QwenVLChatTemplateWrapper` 在将用输入的 prompt 转化为 input prompt 的时候调用了 [`Qwen7BChat`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py) 这个对话模版，如果 LLM-VL 模型的对话模版和对应的 LLM 模型不一致时，如 Yi-VL 和 Yi 模型，那么还需要添加 VLM 模型的对话模版。

## FAQ

- **如何支持 liuhaotian/llava-v1.6-mistral-7b ?**

目前 VLM 中的 LLM 部分仅支持 TurboMind 作为推理引擎，而 TuroboMind 目前不支持推理 mistral 系列模型，后续 pytorch 引擎支持多模态输入或者 TurboMind 支持 mistral 系列模型时可以支持。

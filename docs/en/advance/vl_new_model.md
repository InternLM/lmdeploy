# How to support new model in lmdeploy.vl

Currently, there is a group of VLM models adopting the architecture as shown in the diagram below. Images are processed by the Vision Encoder to obtain image features, which are then projected into the text feature space through Projection. Finally, the concatenated features of images and texts are fed into LLM for inference. One characteristic of such VLM models is that when concatenated features are fed into LLM for inference, there is no distinction between the types of features, and there is no interaction computation between the two types of features.

![arch](https://llava-vl.github.io/images/llava_arch.png)

For models with this type of architecture, it is very convenient to add new model support with LMDeploy.

## Support New Model

The overall process of VLM is as follows, where `ImageEncoder` is responsible for collecting and batching images from the request. The `VisionModel` extracts features. `VLChatTemplateWrapper` is responsible for converting the user input prompt and images into `token_ids` and `embedding position`, and finally LLM utilizes this information for inference. In general, to support a new model, users need to implement two parts:

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
> Typically, VLM models have a corresponding LLM model that doesn't take image inputs, such as Qwen-VL-Chat and Qwen-7B-Chat. Please ensure that the LLM model can be inferred by the TurboMind backend, or that its model structure is identical to the model structures supported by TurboMind.
>
> For what models that TurboMind supports, please refer [this](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/turbomind/supported_models.py)

We will take Qwen/Qwen-VL-Chat model as an example to show how to add support for such models.

### VisonModel

The vision model is responsible for extracting image features, and thanks to the widespread use of huggingface, most models have transformers version. With the api of transformers, we can only load the vision part of the VLM model and use the model's own functions to infer images.

Adding a new vision model mainly requires modification in two places:

1. Extracting the vision model corresponding to the VLM model and implementing the `forward` function for feature extraction.
2. Modifying `load_vl_model` function so that the VLM model can find the corresponding VisionModel.

Below is the VisonModel of Qwen/Qwen-VL-Chat

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

In order to correctly locate `QwenVisionModel` when loading the Qwen/Qwen-VL-Chat model, we should modify the [`load_vl_model`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/model/builder.py) function. Here we use `architectures` of Qwen/Qwen-VL-Chat for identification.

### VLChatTemplateWrapper

The purpose of `VLChatTemplateWrapper` is to insert special tokens that representing the images into the prompt, so that the insertion position of the visual features can be obtained when the prompt is converted into token_ids. For supporting new models, users need to override the `append_image_token` function according to the VLM model.

Adding a new VLChatTemplateWrapper mainly requires modifications in two places:

1. Insert the special tokens representing the images into the prompt according to the chat template of the VLM model.
2. Modify `get_vl_prompt_template` function so that the VLM model can load the corresponding VLChatTemplateWrapper.

Below is the VLChatTemplateWrapper of Qwen/Qwen-VL-Chat.

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

In order to correctly locate the `VLChatTemplateWrapper` when loading the Qwen/Qwen-VL-Chat model, we should modify the [`get_vl_prompt_template`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/vl/templates.py) function. Here we use `architectures` of Qwen/Qwen-VL-Chat for identification.

> \[!NOTE\]
>
> The `QwenVLChatTemplateWrapper` uses the [`Qwen7BChat`](https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/model.py) when converting the user input prompt into the input prompt. If the chat template of the LLM-VL model is different from the corresponding LLM model, such as Yi-VL and Yi models, you should also add the chat template of the VLM model.

## FAQ

- **How to support liuhaotian/llava-v1.6-mistral-7b ?**

Currently, the LLM part of VLM only supports TurboMind as the inference backend, while TurboMind does not support Mistral series models currently. The support may be available in the future when the PyTorch engine supports multi-modal inputs or when TurboMind supports Mistral series models.

# 自定义对话模板

被应用的对话模板效果，可以通过设置日志等级为`INFO`进行观测。

LMDeploy 支持两种添加对话模板的形式：

- 一种是利用现有对话模板，直接配置一个如下的 json 文件使用。

  ```json
  {
      "model_name": "your awesome chat template name",
      "system": "<|im_start|>system\n",
      "meta_instruction": "You are a robot developed by LMDeploy.",
      "eosys": "<|im_end|>\n",
      "user": "<|im_start|>user\n",
      "eoh": "<|im_end|>\n",
      "assistant": "<|im_start|>assistant\n",
      "eoa": "<|im_end|>",
      "separator": "\n",
      "capability": "chat",
      "stop_words": ["<|im_end|>"]
  }
  ```

  model_name 为必填项，可以是 LMDeploy 内置对话模板名（通过 `lmdeploy list` 可查阅），也可以是新名字。其他字段可选填。
  当 model_name 是内置对话模板名时，json文件中各非 null 字段会覆盖原有对话模板的对应属性。
  而当 model_name 是新名字时，它会把将`BaseChatTemplate`直接注册成新的对话模板。其具体定义可以参考[BaseChatTemplate](https://github.com/InternLM/lmdeploy/blob/24bd4b9ab6a15b3952e62bcfc72eaba03bce9dcb/lmdeploy/model.py#L113-L188)。

  这样一个模板将会以下面的形式进行拼接。

  ```
  {system}{meta_instruction}{eosys}{user}{user_content}{eoh}{assistant}{assistant_content}{eoa}{separator}{user}...
  ```

  在使用 CLI 工具时，可以通过 `--chat-template` 传入自定义对话模板，比如：

  ```shell
  lmdeploy serve api_server internlm/internlm2-chat-7b --chat-template ${JSON_FILE}
  ```

  也可以在通过接口函数传入，比如：

  ```python
  from lmdeploy import ChatTemplateConfig, serve

  serve('internlm/internlm2-chat-7b',
        chat_template_config=ChatTemplateConfig.from_json('${JSON_FILE}'))
  ```

- 一种是以 LMDeploy 现有对话模板，自定义一个python对话模板类，注册成功后直接用即可。优点是自定义程度高，可控性强。
  下面是一个注册 LMDeploy 对话模板的例子：

  ```python
  from lmdeploy.model import MODELS, BaseChatTemplate


  @MODELS.register_module(name='customized_model')
  class CustomizedModel(BaseChatTemplate):
      """A customized chat template."""

      def __init__(self,
                   system='<|im_start|>system\n',
                   meta_instruction='You are a robot developed by LMDeploy.',
                   user='<|im_start|>user\n',
                   assistant='<|im_start|>assistant\n',
                   eosys='<|im_end|>\n',
                   eoh='<|im_end|>\n',
                   eoa='<|im_end|>',
                   separator='\n',
                   stop_words=['<|im_end|>', '<|action_end|>']):
          super().__init__(system=system,
                           meta_instruction=meta_instruction,
                           eosys=eosys,
                           user=user,
                           eoh=eoh,
                           assistant=assistant,
                           eoa=eoa,
                           separator=separator,
                           stop_words=stop_words)


  from lmdeploy import ChatTemplateConfig, pipeline

  messages = [{'role': 'user', 'content': 'who are you?'}]
  pipe = pipeline('internlm/internlm2-chat-7b',
                  chat_template_config=ChatTemplateConfig('customized_model'))
  for response in pipe.stream_infer(messages):
      print(response.text, end='')
  ```

  在这个例子中，我们注册了一个 LMDeploy 的对话模板，该模板将模型设置为由 LMDeploy 创造，所以当用户提问模型是谁的时候，模型就会回答由 LMDeploy 所创。

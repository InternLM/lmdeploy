# Customized chat template

The effect of the applied chat template can be observed by **setting log level** `INFO`.

LMDeploy supports two methods of adding chat templates:

- One approach is to utilize an existing conversation template by directly configuring a JSON file like the following.

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

  The null values will be assigned using the default method. `model_name` is a must for the json file. It could be in registered model list (check through `lmdeploy list`). Or, it could be a
  new name. The new name will be registered using `BaseChatTemplate`. The detailed definition is [here](https://github.com/InternLM/lmdeploy/blob/24bd4b9ab6a15b3952e62bcfc72eaba03bce9dcb/lmdeploy/model.py#L113-L188). The new chat template would be like this:

  ```
  {system}{meta_instruction}{eosys}{user}{user_content}{eoh}{assistant}{assistant_content}{eoa}{separator}{user}...
  ```

  You can then pass the file path through the command line.

  ```shell
  lmdeploy serve api_server internlm/internlm2-chat-7b --chat-template ${JSON_FILE}
  ```

  It can also be started through a Python script:

  ```python
  from lmdeploy import ChatTemplateConfig, serve
  serve('internlm/internlm2-chat-7b',
        chat_template_config=ChatTemplateConfig.from_json('${JSON_FILE}'))
  ```

- Another approach is to customize a Python dialogue template class like the existing LMDeploy dialogue templates. It can be used directly after successful registration. The advantages are a high degree of customization and strong controllability. Below is an example of registering an LMDeploy dialogue template.

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

  In this example, we register a LMDeploy dialogue template that sets the model to be created by LMDeploy, so when the user asks who the model is, the model will answer that it was created by LMDeploy.

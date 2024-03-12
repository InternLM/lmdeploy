# Customized chat template

The effect of the applied chat template can be observed by **setting log level** `INFO`.

LMDeploy supports two methods of adding chat templates:

- The first approach is to customize a Python dialogue template class like the existing LMDeploy dialogue templates. It can be used directly after successful registration. The advantages are a high degree of customization and strong controllability. Below is an example of registering an LMDeploy dialogue template.

  ```python
  from typing import Dict, Union

  from lmdeploy import ChatTemplateConfig, serve
  from lmdeploy.model import MODELS, BaseChatTemplate


  @MODELS.register_module(name='customized_model')
  class CustomizedModel(BaseChatTemplate):
      """A customized chat template."""
      def __init__(self, meta_instruction='This is a fake meta instruction.'):
          super().__init__(meta_instruction=meta_instruction)

      def messages2prompt(self,
                          messages: Union[str, Dict],
                          sequence_start: bool = True) -> str:
          """This func apply chat template for input messages
          Args:
              messages (str | Dict): input messages. Could be a str prompt or
                  OpenAI format chat history. The former is for interactive chat.
              sequence_start (bool): Only for interactive chatting. Begin of the
                  prompt token will be removed in interactive chatting when
                  the sequence_start is False.
          Returns:
              string. The return value will be sent to tokenizer.encode directly.
          """
          if isinstance(messages, str):
              return self.get_prompt(messages, sequence_start)
          box_map = dict(user=self.user,
                      assistant=self.assistant,
                      system=self.system)
          eox_map = dict(user=self.eoh,
                      assistant=self.eoa + self.separator,
                      system=self.eosys)
          ret = ''
          for message in messages:
              role = message['role']
              content = message['content']
              ret += f'{box_map[role]}{content}{eox_map[role]}'
          ret += f'{self.assistant}'
          print(f'The applied template result: {ret}')
          return ret  # just a dummpy conversion.


  client = serve('internlm/internlm2-chat-7b',
                chat_template_config=ChatTemplateConfig('customized_model'))
  for item in client.chat_completions_v1('customized_model', [{
          'role': 'user',
          'content': 'hi'
  }]):
      print(item)
  ```

  In this example, we registered an LMDeploy dialogue template that simply returns the input prompt as is, or converts the dialogue history into a string directly. The user needs to implement the actual dialogue template logic themselves, ideally considering both input scenarios. With such a service started, all interfaces can be used.

- Another approach is to utilize an existing conversation template by directly configuring a JSON file like the following.

  ```json
  {
      "model_name": "internlm2-chat-7b",
      "system": null,
      "meta_instruction": "This is a fake meta instruction.",
      "eosys": null,
      "user": null,
      "eoh": null,
      "assistant": null,
      "eoa": null,
      "capability": null
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

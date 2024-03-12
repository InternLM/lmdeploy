# 自定义对话模板

被应用的对话模板效果，可以通过设置日志等级为`INFO`进行观测。

LMDeploy 支持两种添加对话模板的形式：

- 一种是以 LMDeploy 现有对话模板，自定义一个python对话模板类，注册成功后直接用即可。优点是自定义程度高，可控性强。
  下面是一个注册 LMDeploy 对话模板的例子：

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
          if self.meta_instruction is not None:
              ret += f'{self.system}{self.meta_instruction}{self.eosys}'
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

  在这个例子中，我们注册了一个 LMDeploy 的对话模板，该模板只是将输入的 prompt 直接返回，或者
  将对话历史直接转成了一个字符串。用户真正需要的对话模板逻辑，需要用户自己做填充，最好对两种输入情况都考虑到。
  这样启动的服务，各个接口都可以使用。

- 另一种是利用现有对话模板，直接配置一个如下的 json 文件使用。

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
      "separator": null,
      "capability": null
  }
  ```

  其中 null 值将采用对话模板的 default 赋值。而 model_name 是必须要传入的，可以是已有的对话模板名（通过`lmdeploy list`获取），也可以是新的名字。
  新名字会将`BaseChatTemplate`直接注册成新的对话模板。其具体定义可以参考[BaseChatTemplate](https://github.com/InternLM/lmdeploy/blob/24bd4b9ab6a15b3952e62bcfc72eaba03bce9dcb/lmdeploy/model.py#L113-L188)。
  这样一个模板将会以下面的形式进行拼接。

  ```
  {system}{meta_instruction}{eosys}{user}{user_content}{eoh}{assistant}{assistant_content}{eoa}{separator}{user}...
  ```

  可以通过命令行将json文件路径传入。

  ```shell
  lmdeploy serve api_server internlm/internlm2-chat-7b --chat-template ${JSON_FILE}
  ```

  也可以通过 python 脚本启动：

  ```python
  from lmdeploy import ChatTemplateConfig, serve

  serve('internlm/internlm2-chat-7b',
        chat_template_config=ChatTemplateConfig.from_json('${JSON_FILE}'))
  ```

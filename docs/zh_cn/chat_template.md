# 对话模板

在与对话模型聊天时，输入给模型的 prompt，通常并不是用户原始的 prompt，而是按照一个样式，对原始 prompt 进行装饰，才得到符合模型的prompt。这个样式被称为对话提示模板（chat prompt template)，简称为对话模板。

以 InternLM 对话模型为例，当问它"你叫什么名字？"时，实际要发给模型的prompt是：

```text
<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.

<|User|>:你叫什么名字？
<|Bot|>:
```

从这点来说，LMDeploy 的对话模板与 longchain 的对话模板概念是一致的。

不过，除了 prompt，LMDeploy 也把和 token 生成相关的超参放到了对话模板中。这么做的出发点是，我们认为，这些参数和对话模板一样，也是模型属性的一部分。
比如说，模型支持的最大上下文长度 `session_len`，对话停止符 `stop_words`，合适的采样参数 `top_p`，`top_k`, `temperature` 等等。
另外，这些参数和对话模板一样，也是模型交付内容中的一部分。

所以，如果没有特殊说明，本文所指的对话模板均是包括了超参的。

## 对话模板工厂

LMDeploy 采用工厂模式设计和实现对话模板。抽象出对话模板的基类，具体模型的对话模板在派生类中实现。所有类加入到注册表中。在使用时，通过注册表获取类。

```python
from mmengine import Registry
CHAT_TEMPLATES = Registry('templates', locations=['lmdeploy.chat_template'])
```

加入注册表的方法是在类名之上加入`@CHAT_TEMPLATES.register_module(name='xxx')`。 以下是一个例子：

```python
@CHAT_TEMPLATES.register_module(name='llama2')
class Llama2(BaseTemplate):
    ...
```

这里的`name`要全局唯一，因为它是在注册表中索引对话模板的 key。

```{note}
模型转换时，参数 --model-name 要填写的名字必须在对话模板的注册表中
```

从注册表中获取对话模板的方法是：

```python
from lmdeploy.chat_template import CHAT_TEMPLATES
chat_template = CHAT_TEMPLATES.get('llama2')()
```

关于对话模板基类、派生类，将在下文阐述。

## 对话模板抽象

```python
class BaseTemplate:
    """Base chat template."""

    def __init__(self,
                 session_len=2048,
                 top_p=0.8,
                 top_k=None,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 capability='chat',
                 stop_words=None,
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.stop_words = stop_words
        self.capability = capability
```

其中，`session_len` 默认是 2K 长度。`capability`表示模型的能力，默认值是 "chat"，表示对话能力。其他为 token 生成时会用到的超参。

获取由用对话模板装饰过的 prompt，使用基类中的方法 `get_prompt`:

```python
def get_prompt(self, prompt, sequence_start=True):
    """Return the prompt that is concatenated with other elements in the
    chat template.

    Args:
        prompt (str): user's input prompt
        sequence_start (bool): indicator for the first round chat of a
           session sequence
    Returns:
        str: the concatenated prompt
    """
    if self.capability == 'completion':
        return prompt
    else:
        return self.decorate_prompt(prompt, sequence_start)
```

可以看到，当模型能力是"completion"，也就是续写能力时，对话模板不会对用户 prompt 做修饰，直接返回。否则，就用 `decorate_prompt` 对 `prompt` 进行装饰。
这个函数的定义是：

```python
@abstractmethod
def decorate_prompt(self, prompt, sequence_start=True):
    return prompt
```

所有的派生类，都需要继承这个函数。函数中的 `sequence_start`，当它为 True 时，表示对话序列的第一轮对话。

为什么要区分一个对话在序列中是否是首轮呢？主要原因是，LMDeploy 默认采用交互式的推理方式。

这里，插播解释交互式的推理方式，便于大家理解。

我们把用户和AI助手的一个对话序列抽象为：U1A1 U2A2 U3A3 ... UnAn，其中 U 表示User，A 表示 AI Assistant，数字表示第 i 轮对话。

当开始第 i 轮对话时，绝大多数推理引擎，都要求用户侧把前 i-1 轮的对话历史 `U1A1 U2A2 ... Ui-1Ai-1` 和第 i 轮的提示词 `Ui`，一并作为请求发送给推理侧。否则，模型推理时会丧失记忆能力。

而 LMDeploy 只要求把 `Ui` 发给推理侧即可，**不要求** 用户侧发送前 i-1 轮的对话历史。因为这些历史已经被缓存在了推理侧。这种推理方式被成为交互式推理。

通常，在对话序列开始时，AI Assistant会在序列之前加上一些引导信息，比如说，"You are a helpful assistant."。这意味着，在交互式推理模式下，只在第一轮对话中加引导信息，其他轮次，无需添加。所以，我们使用`sequence_start`来区分。

除了 `decorate_prompt`，`BaseTemplate`还定义了另外一个虚函数：

```python
@abstractmethod
def messages2prompt(self, messages, sequence_start=True):
    """Return the prompt that is concatenated with other elements in the
    chat template. When messages arg is a string, return
    self.get_prompt(messages). When messages arg is a chat history, return
    translated prompt from chat history.

    Args:
        messages (str | List): user's input prompt
    Returns:
        str: the concatenated prompt
    """
    if isinstance(messages, str):
        return self.get_prompt(messages)
```

这个函数是专门为 api_server 设计的。api_server 提供了兼容 openai API 的接口：`v1/chat/compeltion`, `v1/completion`。而这两个接口要求用户侧发历史对话，也就是函数中的参数`messages`。
`messages`类型为 str 或者 list，当是 list 时，数据样式和 openai 定义的一致：

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello!"
  },
  ...
]
```

因为 api_server 同样也支持交互式推理，所以函数中加了参数 `sequence_start`。

在接下来的章节中，我们以书生.浦语模型（InternLM）为例，详细讲述如何实现一个模型的对话模板。对于其他 LMDeploy 已经支持的对话模板，请参考 chat_template.py

## 书生.浦语（InternLM）对话模板

InternLM 已开源7B、20B的基座模型，以及对话模型。所有的基座模型没有对话能力。所有的对话模型在 prompt 的装饰方法上，是相同的。仅在部分超参上有区别。

### 基座模型 InternLM-7B

它只有续写能力，超参完全和`BaseTemplate`一致。 所以，只需在`class BaseTemplate`前，增加装饰器 `@CHAT_TEMPLATES.register_module(name='internlm-7b')`，就可以在注册表中就加入 "internlm-7b" 条目

### 基座模型 InternLM-20B

同样是只有续写能力。但区别于 InternLM-7B，它的上下文长度可以到 4096。所以，我们从 `BaseTemplate` 派生子类 `InternLM20B`，把 `session_len` 的缺省值改成 4096。然后，通过装饰器`@CHAT_TEMPLATES.register_module`，把'internlm-20b' 加入到注册表中

```python
@CHAT_TEMPLATES.register_module(name='internlm-20b')
class InternLM20B(BaseTemplate):
    """Generation parameters of InternLM-20B-Base model."""

    def __init__(self, session_len=4096, capability='completion', **kwargs):
        super().__init__(session_len=session_len,
                         capability=capability,
                         **kwargs)
```

### 对话模型 InternLM-Chat-7B

在多轮对话中，InternLM 对话模型的模板是：

```text
<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.

<|User|>:{U1}
<|Bot|>:{A1}
<|User|>:{U2}
<|Bot|>:{A2}
...
<|User|>:{Un}
<|Bot|>:{An}
```

模板中有3个角色：system，user，assistant。它们的值分别是"\<|System|>:"，”\<|User|>:“，”\<|Bot|>:“

system 后，会有一段引导信息。我们称之为 meta_instruction。引导信息和对话之间要换行。可以把这个模式写成：`{system}{meta_instruction}{\n}`

user 的 prompt 后，要换行，可以用`{user}{prompt}{\n}`表示。

assistant 之后是 AI 要生成的答案。

分析出这些信息后，我们就可以定义出对话模板：

```python
class InternLMChat7B(BaseTemplate):
    """Chat template of InternLM model."""

    def __init__(
            self,
            system='<|System|>:',
            meta_instruction="""You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
""",  # noqa: E501
            user='<|User|>:',
            eoh='\n',
            eosys='\n',
            assistant='<|Bot|>:',
            stop_words=['<eoa>'],
            **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eosys = eosys
        self.assistant = assistant
        self.stop_words = stop_words
```

接着，实现虚函数 `decorate_prompt`，装饰用户的 prompt:

```python
def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round chat of a
               session sequence
        Returns:
            str: the concatenated prompt
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.system}{self.meta_instruction}{self.eosys}' \
                   f'{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
        else:
            return f'\n{self.user}{prompt}{self.eoh}' \
                   f'{self.assistant}'
```

从这个函数中，不难发现，在首轮对话时，要把 system、meta_instruction 信息带上。在其他轮次中，`user`之前，加换行，和上轮 `assistant` 的回答区分开。

紧接着，还需要实现另一个虚函数 `messages2prompt`:

```python
def messages2prompt(self, messages, sequence_start=True):
    """Return the prompt that is concatenated with other elements in the
    chat template.

    Args:
        messages (str | List): user's input prompt
        sequence_start (bool): flag to start the sequence
    Returns:
        str: the concatenated prompt
    """
    if isinstance(messages, str):
        return self.get_prompt(messages, sequence_start)
    eox_map = dict(user=self.eoh, assistant=self.eoa, system=self.eosys)
    ret = ''
    if self.meta_instruction:
        ret += f'{self.system}{self.meta_instruction}{self.eosys}'

    for message in messages:
        role = message['role']
        content = message['content']
        ret += f'{eval(f"self.{role}")}{content}{eox_map[role]}'
    ret += f'{self.assistant}'
    return ret
```

最后，需要把对话模板加入到注册表中。在类名之前加上如下的装饰器即可：

```python
@CHAT_TEMPLATES.register_module(name='internlm-chat-7b')
```

至此，InternLM-Chat-7B模型的完整对话模板就实现完了。

另外两个模型 InternLM-Chat-7B-8K、InternLM-Chat-20B，它们的对话模板和 InternLM-Chat-7B 一致，只在超参 `session_len` 上有区别。所以，用了以下的实现方式：

```python
@CHAT_TEMPLATES.register_module(name='internlm-chat-20b')
@CHAT_TEMPLATES.register_module(name='internlm-chat-7b-8k')
class InternLMChat7B8K(InternLMChat7B):
    """Chat template and generation parameters of InternLM-Chat-7B-8K and
    InternLM-Chat-20B models."""

    def __init__(self, session_len=8192, **kwargs):
        super(InternLMChat7B8K, self).__init__(**kwargs)
        self.session_len = session_len
```

## 添加对话模板

总结来说， 对于基座模型，可以参考前文 InternLM-7B、InternLM-20B 的例子，更新相关的超参即可。
对于对话模型，一般模型会在文档或者代码中提供对话模板说明。像前文中 InternLM-Chat-7B 的对话模板那样，

- 首先，要分析模板中的角色、角色对应的内容以及角色之间的拼接关系。
- 然后，按照下面的代码模板，把角色在`__init__`函数中定义好
- 然后，重写两个虚函数，实现角色之间的拼接逻辑
- 最后，给模板起个朗朗上口的名字，把类注册到注册表中

```python
@CHAT_TEMPLATES.register_module(name='an-awsome-name')
class YourAwesomeMModel(BaseTemplate):
    def __init__(self, **kwargs):
        super(YourAwesomeMModel, self).__init__(**kwargs)
        # code block for you template

    def decorate_prompt(self, prompt, sequence_start=True):
        # code block to decorate user's prompt according to the model's
        # chat template
        pass

    def messages2prompt(self, messages, sequence_start=True):
        # code block to stitch messages into one prompt according to
        # the model's chat template
        pass
```

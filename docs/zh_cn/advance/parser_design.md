# Parser 设计

本文介绍 LMDeploy 对话服务路径中的内部 Parser 架构，面向需要新增或审核
reasoning、工具调用协议的贡献者。面向用户的配置方法和请求示例，请参阅
[推理输出](../llm/api_server_reasoning.md)和
[工具调用](../llm/api_server_tools.md)。

Parser 层遵循一条核心原则：

> 识别协议边界，保留模型已经稳定的输出，并在输出契约允许时尽早发出归一化
> delta。

它不会充当模型生成 JSON 或 XML 的严格校验器。重复参数、不完整值以及其他语义
上可疑的输出是否可接受，仍由调用者决定。

## 整体架构

服务路径将传输、响应路由和模型特有语法分为不同层次：

```text
引擎输出 chunk
        |
        v
ChatRunner                         请求生命周期及传输元数据
        |
        v
ResponseParser                    plain/reasoning/tool 路由
        |                         以及响应中尚未消费的后缀
        +--> ReasoningParser      reasoning 标签和初始模式
        |
        +--> ToolParser           单个外层工具块及归一化调用
                  |
                  +--> JSON、类 XML、DSML 或模型特有的 consumer
        |
        v
OpenAI / Anthropic 响应适配层
```

各组件的权责边界如下：

| 组件               | 负责                                                                   | 不负责                    |
| ------------------ | ---------------------------------------------------------------------- | ------------------------- |
| `ChatRunner`       | 迭代引擎、管理请求生命周期、结束状态、token ID 和 log probability      | 模型协议语法              |
| `ResponseParser`   | 在普通内容、reasoning 和工具块之间路由；持有响应中尚未消费的后缀       | 工具 payload 的语法       |
| `ReasoningParser`  | 声明 reasoning 起止标签以及解析是否从 reasoning 模式开始               | 流式 buffer 和响应路由    |
| `ToolParser`       | 管理外层工具块、公共的调用身份及过滤规则，并输出归一化 `DeltaToolCall` | 普通内容或 reasoning 内容 |
| 具体工具 Parser    | 内部 payload 语法及其增量状态                                          | SSE 封装和 API 传输元数据 |
| `JsonValueScanner` | 定位单个 JSON 值的词法结束边界                                         | JSON 语法或 schema 校验   |

对应实现的文件分布如下：

| 区域                        | 主要文件                                                      |
| --------------------------- | ------------------------------------------------------------- |
| 传输层集成                  | `lmdeploy/serve/core/chat_runner.py`                          |
| 响应路由                    | `lmdeploy/serve/parsers/response_parser.py`                   |
| Reasoning 协议声明          | `lmdeploy/serve/parsers/reasoning_parser/reasoning_parser.py` |
| 公共工具调用契约            | `lmdeploy/serve/parsers/tool_parser/tool_parser.py`           |
| JSON envelope 和 value 扫描 | `json_tool_parser.py`、`json_value_scanner.py`                |
| 共享类 XML 状态机           | `xml_tool_parser.py`                                          |

大多数模型使用 `BaseResponseParser`，并为它配置注册过的 reasoning 和工具
Parser。`ResponseParserManager`、`ReasoningParserManager` 和
`ToolParserManager` 负责解析配置的名称。若某种协议的 token channel 语义无法由
这种组合表达，则可以注册专用的 `ResponseParser`；OpenAI Harmony 是目前的
例子。

Parser 对象是有状态的，每个响应拥有独立实例。不要在并发请求或前后两个请求间
共享同一个实例。

## Response Parser

`BaseResponseParser` 持有顶层状态机：

```text
                       reasoning close
                 +---------------------------+
                 |                           v
plain -- reasoning open --> reasoning      plain
  |                              |
  +--------- tool open ----------+
                 |
                 v
                tool -- 外层工具块结束 --> plain
```

引擎 chunk 不需要与这些边界对齐。例如，一个 chunk 可以同时包含 reasoning 的
结束、普通正文、工具起始标签以及部分工具 payload。因此：

```python
ResponseParser.stream_chunk(...) -> list[tuple[DeltaMessage, bool]]
```

可以针对一个引擎 chunk 返回多个消息；当可能的标签或 payload 片段还在缓存时，
它也可以返回空列表。

每个 `DeltaMessage` 携带的布尔值表示该消息是否发出了工具调用。`ChatRunner` 用它
记录本次响应是否产生过工具调用，并据此决定是否将最终的 `stop` 转换为客户端看到
的 `tool_calls`。

`ReasoningParser` 虽然名为 Parser，却有意只充当协议描述器：它声明 reasoning
标签和初始模式；buffer、标签识别以及 channel 路由都由 `BaseResponseParser`
负责。

`BaseResponseParser` 只维护 `self._pending`，即语义尚未稳定的响应后缀。在
plain 和 reasoning 模式下，它通常是某个起止标签的真前缀；在 tool 模式下，它
是 `ToolParser.feed_tool_block` 尚未消费的后缀。

当一个引擎 chunk 被拆成多个可见的 Parser delta 时，`ChatRunner` 只把 token
ID、log probability 和结束状态附加到最后一个 delta，避免传输元数据重复。

### 完整响应与流式响应

完整解析并没有另一套语法实现。`parse_complete` 将全部文本送入
`stream_chunk(..., final=True)`，再分别合并正文、reasoning 片段，并从工具
delta 构造完整调用。因此，边界处理的修复会同时作用于流式和非流式请求。

`final=True` 表示之后不会再有解码文本。Parser 此时会释放原本仍有歧义的普通
文本，并尽力收尾工具调用；这不保证把模型生成的畸形输出修复为合法结果。例如，
类 XML Parser 会闭合已经发出的 OpenAI `function.arguments` 对象，但不会凭空
补出缺失的函数名或参数值。

### 工具块之间的分隔符

一个工具块关闭后，只有确认后续紧接着另一个工具起始标签，Response Parser 才
会忽略中间的换行。因此，以下两种序列的分隔符都不会作为 assistant content
返回：

```text
</tool_call>\n<tool_call>
</tool_call>\n\n<tool_call>
```

如果换行之后是普通文本，或者换行位于整个响应末尾，它仍然属于 content。这个
判断跨越两个外层工具块，应由 Response Parser 负责；单个 Tool Parser 不应消费
它。

## 已消费前缀接口

Response Parser 消费工具起始标签后，会调用：

```python
consumed = tool_parser.feed_tool_block(text, deltas, final=final)
```

`text` 从外层起始标签之后开始，可能同时包含上次保留的文本、新解码文本、外层
结束标签和后续普通正文。该方法追加零个或多个归一化工具 delta，并返回可以安全
丢弃的前导字符数：

```python
pending = text[consumed:]
```

该接口选择返回“已消费前缀”，而不是让 Tool Parser 再持有一份响应 buffer。这样
每层只有一个 buffer owner，Tool Parser 可以精确停在后续正文之前，也不会重复
扫描已经发出的文本。

具体 Parser 实现：

```python
_consume_stream_payload(text, deltas, final=final) -> int
```

它只消费具体协议的内部 payload，不得消费外层结束标签。内部语法结束时，它设置
`_payload_closed`；随后由 `ToolParser.feed_tool_block` 消费外层分隔符，并把
剩余文本交还 Response Parser。

返回 `0`，或者只消费语法而没有发出 delta，都是正常情况。这表示还需要更多解码
文本，才能确认边界或得到可对外发送的稳定片段。

### 两种结束状态

`ToolParser` 有意区分两个结束状态：

| 状态              | 含义                                                    | 所有者           |
| ----------------- | ------------------------------------------------------- | ---------------- |
| `_payload_closed` | 模型特有的内部 payload 已结束；外层结束标签可能尚未到达 | 具体 Tool Parser |
| `block_closed`    | 整个外层工具块已经消费，或最终输入迫使其结束            | `ToolParser`     |

这种区分避免内部 Parser 感知 Response Parser 将如何处理外层分隔符和后续正文。

## 归一化工具调用输出

所有 Tool Parser 都通过 `_begin_call` 和 `_emit_delta` 执行一致的客户端输出
契约。

### 调用身份先行

对于每个被接受的调用 index，第一个可见 delta 必须包含 ID、类型和函数名，参数
片段随后发出。例如，即使模型输出：

```json
{"arguments":{"city":"Beijing"},"name":"weather"}
```

归一化后的顺序在概念上仍是：

```text
DeltaToolCall(index=0, id=..., type="function", name="weather")
DeltaToolCall(index=0, arguments="{\"city\":\"Beijing\"}")
```

如果参数先于函数名出现，Parser 会把各参数片段分别保留，等身份 delta 发出后再按
原顺序释放。这个短暂延迟是下游流式 API 的要求：它们会根据第一个事件创建工具
块，之后无法可靠地补救缺失的函数名。

### 过滤与保留

在服务路径中，`adjust_request` 会记录最终请求暴露的工具名。如果模型生成的函数
名不在该集合中，对应调用不会对外发出。即使前面的调用被拒绝，后续被接受的调用
仍会获得连续且从零开始的输出 index。

除了这条运行时边界，Parser 会尽可能保留模型生成的数据。特别是，重复的 JSON
参数 key 或重复的 XML 参数不会被去重；参数片段保持原始顺序，最终参数对象是否
在语义上可接受由调用者决定。

工具调用按顺序产生：具体 Parser 会在开始下一个调用前结束或放弃当前调用。
`_begin_call` 会重置公共的单次调用状态，因此具体 Parser 不需要为不存在的交错
调用片段维护 index 映射。

## JSON 工具 payload

`JsonToolParser` 负责包含函数名和参数字段的 JSON envelope。Qwen Parser 使用
`arguments`，InternLM 和 Llama Parser 则将 `argument_field` 覆盖为
`parameters`。派生类通常只需声明外层标签和参数字段名。

Envelope Parser 会增量识别 key、冒号、函数名、value 和对象结尾。参数 value
中的源文本一旦稳定便原样发出；其他 value 只扫描到足以定位下一个 envelope 字段
的位置。如果 envelope 没有参数字段，Parser 只发出一次 `{}`。

`JsonValueScanner` 是词法扫描器。它跟踪容器深度、字符串和转义，但有意避免反
序列化再序列化参数值。这带来三个好处：

- 保留重复 key 和原始格式；
- 大 value 无需等到完整缓存后再创建第二份副本；
- 已消费的输入不会被重复搜索，整体工作量接近线性。

长字符串和容器使用批量语法搜索，极短后缀使用直接循环。不要改为反复从 pending
buffer 开头重新搜索。

## 类 XML 工具 payload

`XmlToolParser` 是顺序、函数先行协议的共享驱动器，这类协议的参数 value 以具体
方言的结束标签终止；GLM-4.7 和 Qwen3-Coder 是两个例子。其状态迁移如下：

```text
function --> arg_start --> arg_name --> arg_value
                ^                         |
                +-------------------------+
                |
                +---- payload end ----> done
```

只有基类持有 `XmlParseState`。具体方言通过三个 hook 识别语法并返回语义结果，
不直接修改共享状态：

```python
_consume_function(payload, pos, final) -> (pos, name, "arg_start" | "done") | None
_consume_arg_start(payload, pos) -> (pos, "arg_name" | "done") | None
_consume_arg_name(payload, pos) -> (pos, argument_name) | None
```

`None` 表示尚未消费的后缀不完整，需要继续保留。共享 driver 负责应用成功的状态
迁移。`XmlToolParser` 自身负责消费参数 value、发出 JSON 对象、根据 schema 做
标量类型转换，以及迁移回 `arg_start`。

该边界有意比“所有 XML 外形协议”更窄。参数可能先于函数身份出现的格式不符合
`XmlToolParser` 契约，因为查询参数 schema 需要先知道函数名。参数 value 不是由
结束标签分隔的格式，也应该直接继承 `ToolParser`，或者使用其他合适的共享
Parser。

类型明确后，未加引号的普通字符串可以立即流式发出。带引号的 value 和 schema
声明为非字符串的 value 需要缓存到结束标签，再统一归一化和序列化。Parser 最多
只保留用于完成判断的语法，以及本质上需要完整转换的 value。

## 与 token 边界对齐的标签前缀

解码 chunk 终止在 token 边界，而不是任意字符位置。因此，Parser 只应保留确实
可能出现在解码 chunk 尾部的标签真前缀。

例如，如果 `</parameter>` 最多被编码为 `</`、`parameter`、`>` 三个 token，
那么可能出现的不完整后缀只有：

```python
arg_value_close_prefixes = ("</parameter", "</")
```

无需测试 `<`、`</p` 或 `</para` 等每一个字符前缀。如果支持的 tokenizer 将整个
标签编码成一个 token，则前缀 tuple 为空。`_stable_prefix_end` 使用按长度从长到
短排列的 tuple，在热路径中无需重新 tokenize 就能返回安全的可消费边界。

必须针对该 Parser 支持的每一个 tokenizer 验证这个 tuple。任意字符分块适合一般
JSON scanner 测试，但不能正确模拟原子或多 token 的协议标签。

## 新增 Parser

根据模型协议选择边界最窄、能够满足要求的已有抽象：

1. JSON envelope 仅改变外层标签或参数字段名时，继承 `JsonToolParser`；
2. 只有顺序、函数先行且参数由结束标签分隔的协议，才继承 `XmlToolParser`；
3. DSML 或 section/call 等明显不同的语法，直接继承 `ToolParser`；
4. 只有顶层响应 channel 或 token 语义无法由 `BaseResponseParser` 组合表达时，
   才新增专用 `ResponseParser`。

一个最小 JSON 方言如下：

```python
from .json_tool_parser import JsonToolParser
from .tool_parser import ToolParserManager


@ToolParserManager.register_module("example-json")
class ExampleJsonToolParser(JsonToolParser):
    argument_field = "arguments"

    @classmethod
    def get_tool_open_tag(cls) -> str:
        return "<tool_call>"

    @classmethod
    def get_tool_close_tag(cls) -> str:
        return "</tool_call>"
```

还需从 `tool_parser/__init__.py` 导入该模块，确保注册发现和 CLI 校验可以找到它。
如果结束标签可以在支持的 token 边界处拆分，应通过 `tool_close_prefixes` 声明其
真前缀；不要添加猜测性的字符前缀。

协议特有的 prompt 设置应放在 `adjust_request` 中。始终调用
`super().adjust_request(request)`，保证工具归一化和 allowed-name 过滤仍然执行。
只有当 `structural_tag_model`（以及必要时的 `reasoning_structural_tag_model`）
能够选择匹配的 XGrammar structural-tag 格式时，该 Parser 才支持
`tool_choice="required"`。

## 实现不变量

实现或审核 Parser 时应保持以下不变量：

- 除调用身份或完整 value 转换确实要求缓存外，尽早发出所有稳定片段；
- 每个逻辑调用在发出内容前，恰好调用一次 `_begin_call`；
- 所有客户端可见的调用片段都通过 `_emit_delta` 发出；
- 只消费稳定前缀，绝不消费后续 assistant content；
- 将外层结束标签留给 `ToolParser.feed_tool_block`；
- 内部语法结束后设置 `_payload_closed`；
- 保留片段顺序和重复参数，不在流式热路径增加语义校验；
- 避免对已经消费的文本重复 decode、tokenize、parse 或 scan；
- 将 `final=True` 理解为输入结束，而不是 payload 合法性的证明。

## 测试与 benchmark

每次 Parser 修改都应比较流式重建结果、完整解析结果和预期归一化结果。至少覆盖：

- 整个响应位于一个 chunk；
- 起止标签在所有真实 token 边界处分块；
- 在源格式允许两种顺序时，函数名分别位于参数之前和之后；
- 无参数调用、重复参数、多个顺序调用和未知函数名；
- 最终 chunk 中不完整或畸形的 payload，以及工具块后的普通正文；
- reasoning、content 和工具片段位于同一个引擎 chunk；
- 一个引擎 chunk 产生多个 Parser delta 时，token ID 和 log probability 的传输。

JSON scanner 的性能测试应包含小 chunk 和大 chunk（例如 1、4、8、16、256 个字符）
以及包含大量短字符串的输入。Benchmark fixture 必须使用所选 Parser 的真实语法
（`arguments` 与 `parameters`、XML 标签、DSML 等），并在报告吞吐量前校验重建
结果。

# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) 通过**打分**而不是生成来回答结构化问题。一个 Systemone 请求包含
`state`、最多 16 个 `questions` 和可选图片 `assets`。模型读取一个以 JSON 骨架结尾的提示，骨架中每个问题对应一个 `<decision>`
标记；取每个标记前一位置的 logits，并限定在该问题的答案符号（`A`、`B`……）上，即得到答案分布。

LMDeploy 在 PyTorch 引擎上以 `intern-decision` 任务提供该模型。模型检查点本身是普通的 Qwen3.5 视觉语言模型，任务只是在其上增加协议层。
决策协议与预处理改编自 Apache-2.0 许可的 [Intern-Decision 项目](https://github.com/InternLM/Intern-Decision)，改编文件保留了署名。

当前仅通过 HTTP 接收决策请求，暂不支持 Python Pipeline 的决策入口。
对外 schema 仅保留 `SystemoneRequest` / `SystemoneResponse` 和 `OpenAIDecisionRequest` / `OpenAIDecisionResponse`。
Parser 将 OpenAI 请求转换成私有 `_DecisionRow`，只保存 `state`、`questions` 和 `images`；
不再接受旧原生请求格式或返回旧响应封装。

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision --model-name Intern-Decision-0.8B
```

`--task intern-decision` 只启用决策评分并选择 PyTorch 后端，不再接受旧的模型检查点名称作为任务名。
其余 PyTorch 参数（批大小、TP、前缀缓存等）照常可用，决策请求只做输入评分，不生成 token。

启动时用 AST 尝试读取本地模型检查点的 `inference.py` 中有限且为正的 `DEFAULT_TEMPERATURE` 常量，
不执行该文件。文件或常量缺失、读取/解析失败、数值非法时记录警告，回退到 4B 温度 `1.99241824`。
不为温度单独下载远程模型文件，也不验证温度与权重的对应关系。警告遵循 `--log-level`（命令行默认 `WARNING`）。

只读共享对象 `DecisionContext` 仅保存标记 ID、答案符号 ID、图片 token ID 和温度，复用已有引擎资源初始化一次；
对象存在即表示决策端点已启用。

`POST /v1/decisions` 只接收并返回下文的 OpenAI Decisions 文本/图片协议；OpenAPI 明确声明单一请求模型和单一成功响应模型。
`POST /v1/systemone` 接收并返回下文的 Systemone 协议，请求包含 `state`、`questions` 对象和可选的 `assets`。
向 `/v1/decisions` 发送原生请求会返回 422。
文本和图片评分均可使用 `/v1/systemone`；图片通过 `assets` 上传，并在请求内容中引用。

另见 [HTTP 客户端示例](../../../examples/decision/decision.py)。

## Systemone 图片资源协议

`POST /v1/systemone` 复用 `intern-decision-inference`（a8fe089）的图片资源协议，使用共享的决策引擎。
支持在 JSON 中携带图片资源或通过 `multipart/form-data` 上传文件；`/v1/decisions` 保持原有 OpenAI 协议。

```json
{
  "state": {"photo": "photo-ref"},
  "assets": {"photo-ref": {"format": "png", "data_base64": "<带标准填充的裸 Base64>"}},
  "questions": {"red": {"type": "noul", "instructions": "photo-ref 中是否有红色？"}}
}
```

图片引用会在 `state`、`instructions` 和 `criteria` 的字符串值中原位展开，对象键保持不变。
每个图片资源都必须被引用；最多允许展开 8 次图片引用，重复引用也计入次数。
仅支持静态 PNG/JPEG 文件，每个文件不超过 16 MiB，原始宽高均须在 280–1568 像素之间。
Base64 必须采用带标准填充的规范编码，不能带 data URL 前缀。HTTP 请求体不超过 32 MiB，不接受图片 URL 或本地路径。
使用 multipart 上传时，第一个部分必须是名为 `request`、内容类型为 `application/json` 的请求数据；
图片资源通过 `part` 引用各自独立的文件部分。文件内容优先于 `data_base64`；省略 `format` 时根据文件内容推断格式。

图片处理沿用参考服务的策略：转换为 RGB，不应用 EXIF 旋转，缩放尺寸按 32 像素对齐，
单图视觉 token 预算为 1024，整组图片预算为 4096。图片内容或引用无效时返回 400，请求结构校验失败返回 422，
超过大小限制返回 413。该端点不接受旧的 `images` 字段。

响应包含 `model`、`answers` 和 `usage`；评分答案的 `legend` 保留原始 JSON 值和图片引用。
`usage.input_tokens` 包含视觉 token。`usage.output_tokens` 统计 assistant 答案骨架分词后的 token 数，
其中包含决策标记，不表示实际生成的 token 数。图片仅作为输入依据，不随响应返回。

## OpenAI Decisions 输入

向 `/v1/decisions` 发送 `model`、字符串或消息数组形式的 `input`，以及 `questions` 数组：

```json
{
  "model": "Intern-Decision-0.8B",
  "input": "客户被重复扣费。",
  "questions": [
    {"type": "predicate", "name": "duplicate", "instructions": "客户是否被重复扣费？"},
    {"type": "choice", "name": "team", "instructions": "应该由哪个团队处理？",
     "choices": [{"value": "billing"}, {"value": "support"}]},
    {"type": "score", "name": "severity", "instructions": "评估严重程度。",
     "levels": [{"label": "低"}, {"label": "中"}, {"label": "高"}]}
  ]
}
```

`input` 也接受非空的 `user` 消息数组。每条消息的内容可以是字符串，也可以是非空的
`input_text` / `input_image` 内容块数组；`type: "message"` 可省略。例如：

```json
{
  "model": "Intern-Decision-0.8B",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "检查这盏交通灯。"},
    {"type": "input_image", "image_url": "https://example.com/light.png"}
  ]}],
  "questions": [{"type": "predicate", "name": "red", "instructions": "是否亮红灯？"}]
}
```

图片引用接受 HTTP(S) URL 和内联 `data:image/...;base64,...` URL，遵循公共媒体加载器的 URL 限制和域名白名单，
不接受服务器本地文件路径。图片的 `detail` 可以省略、设为 `null` 或 `"auto"`；其他值会被拒绝。
每个请求最多 8 次图片出现，重复引用也计入。内联和下载后的图片均须为静态 PNG、JPEG、WEBP 或 GIF，
图片文件解码后不超过 10 MiB、2000 万像素、单边 20000 像素，宽高比不超过 200:1。图片内容非法时返回带原字段位置的 422。
HTTP 下载沿用公共加载器的传输行为，下载完成后检查文件字节上限。图片转为 RGB，并按 32 像素对齐尺寸，
沿用 a8fe089 的单图 1024、整组 4096 视觉 token 预算；整组预算按图片出现次数分配。

解析器按图片在所有消息中的出现顺序收集图片，并在内部状态中写入从 1 开始的 `image_index` 引用。
重复引用占据独立的位置。编译器会将实际图片统一放在状态与问题定义文本之前，因此保留的是文本引用关系，
并不保持图文 token 的交错位置。仅含一条字符串内容消息时，提示与直接传入字符串相同，响应映射规则也相同。

当前子集支持 1–16 个问题、字符串形式的题目说明、2–62 个不重复的字符串或布尔选项，以及 2–62 个评分等级。
62 个评分等级是对齐参考部署仓库的本地扩展，OpenAI 官方当前文档的上限仍为 10。
答案按请求顺序以数组返回。题目名称可以省略或为 `null`；显式提供的名称必须非空白且不重复。
未命名时响应中的名称为 `null`。字符串选项值不能为空白，顶层字符串 `input` 不能为空字符串。
校验不会裁剪名称、选项值或输入文本中的空白。
判断题的 `probability` 对应现有 `noul` 值，即 P(yes)。选项值保留原始类型，包括区分布尔值 `true` 和字符串 `"true"`。
评分使用从 0 开始的等级索引，并在响应中恢复原始等级标签。
本地 choice/score 的 `confidence` 是温度校准后的最大候选概率，与 OpenAI 的置信度校准语义不同。

`model` 必填，接受 `/v1/models` 返回的服务模型名、`InternLM/<服务模型名>` 或
`internlm/<转为小写的服务模型名>`，不用于切换权重。
模型身份由 HTTP 端点处理，两种协议共用同一个 `DecisionContext` 和实际读取或回退得到的温度。
可选的 `safety_identifier` 接受最长 64 字符的字符串或 `null`，表示调用方提供的终端用户标识。
本服务仅兼容接收并校验；它不是鉴权凭据，不进入提示，也不影响评分。
其他消息角色、音频/视频/文件内容块、结构化题目说明、工具记录、缓存控制和未知字段均返回 422。
`/v1/systemone` 仅接受 Systemone 协议，混用原生字段与 OpenAI 字段会被拒绝。

OpenAI 格式与 Systemone 格式均将 assistant 答案骨架分词后的 token 数计入 `output_tokens`，
包括决策标记、不包括聊天模板包装，实际不生成 token。
`input_tokens` 保留引擎输入计数（含骨架），不额外扣除骨架；`total_tokens` 为输入与输出计数之和。
这是参考部署仓库的用量统计口径。
`cached_tokens` 来自引擎。**`cache_write_tokens: 0` 是为兼容 SDK 而设置的未统计占位值，不是缓存写入量的测量结果。**
推理 token 数为 0。评分器不输出拒答结果。
两个决策入口的请求结构和模型名校验错误均返回 422，并通过 `detail` 和 `error.details` 指明出错字段。
资产与评分错误保留各自的状态码，其他 API 路由的错误格式保持不变。
当前实现支持 [OpenAI Decisions 协议](https://developers.openai.com/api/docs/guides/decisions)的一个子集，不代表完整 API 等价。

可使用提供 `client.decisions` 的 SDK，并设置 `base_url="http://127.0.0.1:23333/v1"`，也可直接发送 HTTP 请求。

## Systemone 响应与校准

响应仅包含 `model`、`answers` 和 `usage`。choice 和 score 答案包含 `probabilities`、`confidence`，
以及对应的 `choice` 或 `score`/`legend` 字段。noul 答案仅包含 `type` 和 `noul`（P(yes)）。
候选 softmax 与温度校准沿用模型检查点的 `inference.py`；官方 0.8B 检查点的温度为 2.747760550703，
读取失败时使用上文所述的 4B 默认值。响应不返回温度和耗时元数据。
不同推理内核与浮点精度可能使结果与官方 `DecisionEngine` 存在小幅数值差异。
`usage.output_tokens` 统计包含决策标记的 assistant 答案骨架分词后的 token 数，实际不生成 token。

## 运行方式

决策端点直接驱动引擎并持有本次请求的会话。`DecisionParser` 只负责决策协议与引擎输入/输出之间的转换，不执行引擎调度。
不另设 `DecideRunner`、不依赖公共 runner 的扩展钩子，也没有决策专属视觉流程或自行加载资源的 `DecisionEncoder`：

1. `compile_row` 将 `state` 和 `questions` 编译成包含训练时 assistant 答案骨架的消息。
   Systemone 图片引用在原位展开；解析器解码并缩放图片，再将携带 PIL 图片对象的 `image_data` 内容块交给公共预处理。
   OpenAI 图片采用独立校验限制，远程 URL 使用公共加载器下载，再按相同策略 resize 并转换成 `image_data`。
   启动时只使用引擎已有的分词器校验标记及候选 token，保存少量共享元数据，不再加载第二套分词器或处理器。
2. 解析器设置 `do_preprocess=True`、`enable_thinking=False` 和 `add_vision_id=True`，
   使用服务配置的模板，复用普通聊天预处理。请使用模型检查点自带的 HF 模板；该模板会去掉末尾的 assistant 结束符。
   图片通过已加载的处理器进行分词并展开视觉 token。
3. 预处理完成后，解析器在最终输入 token ID 中定位 `<decision>`，把 `logprob_start_len` 设置为首个标记前一位置，
   并保存各问题的评分行索引和预期评分行数，无须提前计算图片展开后的 token 数。
4. `GenerationConfig(max_new_tokens=0, logprobs=0, logprob_extra_token_ids=...)` 请求输入评分。引擎对所选后缀计算 LM head，
   返回指定的 62 个候选分数，候选不受 top-k 排名限制。这里缩减的是返回的分数，不是 LM head 投影的词表维度。
5. 解析器校验返回的评分行数，按保存的行索引取各问题的分数，按各问题的候选集合归一化、应用温度校准并构造响应。
   引擎返回原始 logits 或原始对数概率时，均支持这一步候选归一化。

整个请求只执行预填充，必要时分块处理，不进入自回归解码。调度、会话释放与取消复用
`ServerContext.create_session` 和公共的请求清理函数；前缀缓存复用在需要评分的位置之前停止，保证这些分数重新计算。

两个决策入口的图片准备均在工作线程中执行，图片由请求持有，重复引用复用解码和缩放结果。
准备过程部分失败时关闭已准备的图片。公共预处理消费携带 PIL 图片的内容块并展开视觉 token，不再重复下载或解码。

## 限制

- 仅支持 PyTorch 后端；不支持投机解码与 PD 分离角色。
- 图片展开后必须小于引擎会话长度（即模型自身的上下文长度，除非用 `--session-len` 覆盖）；
  超限输入会被拒绝，不会截断。
- Systemone 请求包含 1–16 个问题，每个 `choice` 或 `score` 问题有 1–62 个选项，按请求顺序分配答案符号，
  响应中恢复原始选项名称。
- Systemone 的 `model` 可以省略或为空字符串，也可填写服务模型名、`InternLM/<服务模型名>` 或
  `internlm/<转为小写的服务模型名>`。如需使用简短名称，请显式设置 `--model-name`。
- `--dp > 1` 时请直接访问各 rank 的服务：DP 代理不转发 `/v1/decisions`。

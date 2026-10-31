# Intern-Decision

[Intern-Decision-0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) 通过**打分**而不是生成来回答结构化问题。一个原生请求包含
`state`、最多 16 个 `questions` 和可选图片。模型读取一个以 JSON 骨架结尾的提示，骨架中每个问题对应一个 `<decision>`
标记；取每个标记前一位置的 logits，并限定在该问题的答案符号（`A`、`B`……）上，即得到答案分布。

LMDeploy 在 PyTorch 引擎上以 `intern-decision` 任务提供该模型。checkpoint 本身是普通的 Qwen3.5 视觉语言模型，任务只是在其上增加协议层。
决策协议与预处理改编自 Apache-2.0 许可的 [Intern-Decision 项目](https://github.com/InternLM/Intern-Decision)，改编文件保留了署名。

当前仅通过 HTTP 接收 Decision 请求，暂不支持 Python Pipeline 的 Decision 入口。

## HTTP

```shell
lmdeploy serve api_server /path/to/Intern-Decision-0.8B --task intern-decision --model-name Intern-Decision-0.8B
```

`--task intern-decision` 只启用决策评分并选择 PyTorch 后端，不再接受旧的 checkpoint 名称作为 task。
其余 PyTorch 参数（批大小、TP、前缀缓存等）照常可用，决策请求只做输入评分，不生成 token。

启动时用 AST 尝试读取本地 checkpoint 的 `inference.py` 中有限且为正的 `DEFAULT_TEMPERATURE` 常量，
不执行该文件。文件或常量缺失、读取/解析失败、数值非法时记录 warning，回退到 4B 温度 `1.99241824`。
不为温度单独下载远程模型文件，也不验证温度与权重的对应关系。warning 遵循 `--log-level`（CLI 默认 `WARNING`）。

只读共享对象 `DecisionContext` 仅保存 marker ID、答案符号 ID、图片 token ID 和温度，复用已有引擎资源初始化一次；
对象存在即表示决策端点已启用。

`POST /v1/decisions` 只接收并返回下文的 OpenAI Decisions 文本/图片协议；OpenAPI 明确声明单一请求模型和单一成功响应模型。
`POST /v1/jev` 只接收并返回原生协议，包含 `state`、`questions` 对象和可选的 `images`。
已有原生客户端请改用 `/v1/jev`；向 `/v1/decisions` 发送原生请求会返回 400。
`images` 复用 `/generate.image_data` 的 `ImageDataFormat`：接受字符串、`{"url": "..."}` 参数字典，或二者混合的列表。
例如 `"images": ["data:image/png;base64,...", {"url": "https://example.com/image.png"}]`；单图也可以直接传字符串或字典。
字符串支持 HTTP(S) URL、完整 base64 data URL 或本地路径。省略 `images` 或传 `[]` 表示纯文本输入。
字典内部字段交给公共多模态处理器检查，额外参数原样传递，其效果以已有处理器为准。
**迁移说明：** 传图片引用，不传完整 `{"type": "image_url", "image_url": {...}}` 消息块；compiler 会统一包装一次。
上游 `{"type": "image/png", "data": "<裸 base64>"}` 上传格式不做兼容转换，上传内容请先编码为完整 data URL。
本地路径指运行 LMDeploy 的机器上的文件。Decision HTTP 请求复用 chat 的多模态加载器，包括 URL 检查与 `allowed_media_domains` 配置。
图片格式、base64 解码和动画处理遵循框架已有行为，不额外设置 Decision 专属的图片张数、上传字节数或原图像素数限制。
checkpoint processor 负责缩放和视觉 token 展开；加载器按存储方向读取像素，不应用 EXIF 旋转。

错误沿用 LMDeploy 的平铺错误体。非法输入返回 400，包括向没有视觉编码器的引擎提交图片；
图片能力由公共多模态预处理器检查。未启用决策任务时返回 503。

另见 [HTTP 客户端示例](../../../examples/decision/decision.py)。

## OpenAI Decisions input

Send `model`, a string or message-array `input`, and a `questions` array to `/v1/decisions`:

```json
{
  "model": "Intern-Decision-0.8B",
  "input": "The customer was charged twice.",
  "questions": [
    {"type": "predicate", "name": "duplicate", "instructions": "Was the customer charged twice?"},
    {"type": "choice", "name": "team", "instructions": "Which team should handle this?",
     "choices": [{"value": "billing"}, {"value": "support"}]},
    {"type": "score", "name": "severity", "instructions": "Rate the severity.",
     "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}]}
  ]
}
```

`input` also accepts a nonempty array of `user` messages. Each message has string content or a nonempty list of
`input_text` / `input_image` blocks; `type: "message"` is optional. For example:

```json
{
  "model": "Intern-Decision-0.8B",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "Inspect this traffic light."},
    {"type": "input_image", "image_url": "https://example.com/light.png"}
  ]}],
  "questions": [{"type": "predicate", "name": "red", "instructions": "Is the light red?"}]
}
```

Image references accept HTTP(S) URLs and inline `data:image/...;base64,...` URLs, using the shared media loader's
URL restrictions. Server file paths remain a native-protocol feature. Image `detail` may be omitted, `null`, or
`"auto"`, which uses the local processor's default policy; other profiles are rejected.

The parser collects images in encounter order across all messages and writes one-based `image_index` references
into the internal state. Repeated references occupy separate positions. The compiler places actual images before
the state/schema text, so this preserves textual references, not interleaved image-token positions. A single message
with string content keeps the same prompt as a plain string input. Response mapping is unchanged.

This subset supports 1–16 questions, string instructions, 2–62 unique string/boolean choices, and 2–10 score levels.
Answers are returned as an array in request order. Names may be omitted, empty, or repeated; omitted names return
`null`. Predicate `probability` is the existing `noul` value, P(yes). Choice values retain their original type, including
the distinction between `true` and `"true"`. Scores use zero-based level indices and restore the original level labels.
Local choice/score `confidence` is the maximum calibrated candidate probability, not OpenAI's confidence calibration.

`model` 必填，必须匹配 `/v1/models` 返回的服务模型名，不用于切换权重。
模型身份由 HTTP 端点处理，两种协议共用同一个 `DecisionContext` 和实际读取或回退得到的温度。 Other message roles, audio/video/file
blocks, structured instructions, tool records, cache controls, and unknown fields are rejected with 400. `/v1/jev` only accepts the native protocol; mixed native/OpenAI fields are rejected.

OpenAI-shaped responses report actual engine input and generated token counts; output tokens are zero and total
tokens are their sum. `cached_tokens` comes from the engine. **`cache_write_tokens: 0` is an unmeasured SDK compatibility
placeholder, not a measurement of cache writes.** Reasoning tokens are zero. Native usage retains its historical
meaning below. The scorer has no refusal output, and errors retain LMDeploy's flat body. This is a supported subset
of the [OpenAI Decisions protocol](https://developers.openai.com/api/docs/guides/decisions), not full API equivalence.

Use an SDK exposing `client.decisions` with `base_url="http://127.0.0.1:23333/v1"`, or call HTTP directly.

## Native response and calibration

每个答案包含 `probabilities`、`confidence`、`decision`，以及按类型给出的 `choice`、`score`（按概率加权的数值期望，附 `legend`）或
`noul`（`yes` 的概率）。候选 softmax 与温度校准算法沿用 checkpoint 的 `inference.py`。
从官方 Intern-Decision-0.8B checkpoint 读取的温度为 2.747760550703；读取失败则使用 4B 默认值。
响应中的 `calibration` 字段注明实际使用的温度。推理内核与浮点精度不同可能带来小幅数值误差，因此不保证与官方
`DecisionEngine` 的输出逐位一致。`usage.output_tokens` 为兼容协议而表示回答的问题数，`usage.generated_tokens` 恒为 0。

## 运行方式

决策端点直接驱动引擎并持有本次请求的 session。`DecisionParser` 只负责决策协议与引擎输入/输出之间的转换，不执行引擎调度。
不另设 `DecideRunner`、不依赖公共 runner 的扩展钩子，也没有决策专属视觉流程或自行加载资源的 `DecisionEncoder`：

1. `compile_row` 将 `state`、`questions` 编译成含训练时 assistant 骨架的 messages，沿用 `/generate` 做法，
   将图片引用包装为 `image_url` 块并放在 user 文本前；
   `DecisionParser` 直接把完整消息交给公共预处理。
   启动时只使用引擎已有 tokenizer 校验标记及候选 token，保存少量共享元数据，不再加载第二套 tokenizer/processor。
2. The parser sets `do_preprocess=True`, `enable_thinking=False` and `add_vision_id=True`, reusing ordinary chat
   preprocessing with the service's configured template. Use the checkpoint's HF template; it trims the final
   assistant terminator. Images use the already loaded processor to tokenize and expand visual tokens.
3. 预处理完成后，parser 在最终 input IDs 中定位 `<decision>`，把 `logprob_start_len` 设置为首个标记前一位置，
   并保存各问题的评分行索引和预期评分行数，无须提前计算图片展开后的 token 数。
4. `GenerationConfig(max_new_tokens=0, logprobs=0, logprob_extra_token_ids=...)` 请求输入评分。引擎对所选后缀计算 LM head，
   返回指定的 62 个候选分数，候选不受 top-k 排名限制。这里缩减的是返回的分数，不是 LM head 投影的词表维度。
5. parser 校验返回评分行数，按保存的行索引取各问题的分数，按各问题的候选集合归一化、应用温度校准并构造响应。
   引擎返回 raw logits 或 raw log probabilities 都支持这一步候选归一化。

整个请求只执行 prefill，必要时分块预填充，不进入自回归 decode。调度、session 释放与取消复用
`ServerContext.create_session` 和公共的请求清理函数；前缀缓存复用在需要评分的位置之前停止，保证这些分数重新计算。

Decision 图片引用包装成消息块后进入公共多模态预处理，包装过程不解码图片。加载器创建的图片在视觉预处理完成后关闭；
部分失败或请求取消时，先等待仍在运行的 worker，再释放图片。

## 限制

- 仅支持 PyTorch 后端；不支持投机解码与 PD 分离角色。
- 图片展开后必须小于引擎会话长度（即模型自身的上下文长度，除非用 `--session-len` 覆盖）；
  超限输入会被拒绝，不会截断。
- Native requests contain 1–16 questions and 1–62 options per `choice` or `score`, assigned symbols in request order.
  响应中恢复原始选项名称。
- 原生请求可省略 `model`；指定时必须匹配服务模型名（`--model-name`，默认是模型路径），不再额外接受 checkpoint 目录名别名。
- `--dp > 1` 时请直接访问各 rank 的服务：DP 代理不转发 `/v1/decisions`。

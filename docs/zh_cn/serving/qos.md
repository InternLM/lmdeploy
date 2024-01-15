## LMDeploy-QoS 介绍与用法

### 背景

在过去一段时间，推理框架伴随着LLM和AGI出现。许多推理框架为语言模型提供可扩展和高性能的在线工作负载服务。它们的工作负载通常涉及多个用户群体，而且工作负载在短时间内快速变化。许多推理框架在满足这些多租户流量模式的要求方面存在困难，而且未能很好的规范约束用户的行为，所以我们认为在LLM推理框架考虑多用户负载均衡是很有必要的。

### 多租户处理的用户分类

LMDeploy-QoS与LMDeploy 提供一系列多租户功能。它要求用户使用适当的用户标识(配置文件或代码库中的user_id)标记其推理请求。它是基于字典的配置作为多租户策略。在这个配置中，用户被映射到不同“用户组”中，并配备一个使用配额。我们的多租户策略可以读取配置，并根据其用户组的优先级和预定义配额与实时分配比率之间的差异安排用户推理请求的调度。经过完备的测试，我们的LMDeploy-QoS模块极大地提高了LLM的服务可靠性并提升了大型语言模型推理工作的GPU资源利用率。

LMDeploy将用户分为4组：

- 白金（Platinum）
- 金（Gold）
- 银（Silver）
- 青铜（Bronze）

根据我们在提供LLM服务方面的使用经验，我们可以将以下4种类型的用户映射到这些用户组中：

- Platinum : VIP用户或管理员用户。包括需要不间断使用的的服务开发人员或演示人员。他们的工作负载频率低，对推理工作的资源需求也不高。

- Gold : 签署定期服务的高级用户，他们需要可衡量的可靠服务。例如，某个公司A与LLM服务提供商签订了合同，购买了每秒X个请求的服务能力，可用性为Z%，供A公司员工使用，年付Y百万美元。

- Silver : 绝大多数用户。大多数试用或每月订阅的用户被归类为此类别。他们需要相对较少的服务，但他们的用户体验对于LLM服务的声誉也很重要。

- Bronze : 支付很少费用给LLM提供商的重度用户。

以上引入用户组分类的目的是为了提供指导，而不是为所有LMDeploy用户提供建议，因为这并不一定适用于所有LLM业务提供商。管理员可以对用户的日常负载进行统计，自行决定如何对用户进行分类。

接下来让我们讨论一下LMDeploy如何根据这些分类进行分配请求。

### 多租户策略

#### 策略 1: 用户组之间的优先级调度

我们引入“用户组”概念。由模块使用者来定义哪些用户到用户组的映射（可以理解为 uid 到用户组的映射）。推荐用户组为4组如下：

- Platinum
- Gold
- Silver
- Bronze

四个用户组之间的优先级顺序是严格的 Platinum > Gold > Silver > Bronze 。当系统繁忙的时候，我们会优先执行排名靠前的请求。

下面的图表显示了优先级处理的工作原理。您可以看到 Platinum 请求已被重新设置优先级并移至队列头部。

![](https://github.com/InternLM/lmdeploy/assets/52888924/9d63f081-7168-4c74-8456-24f0a4b41649)

#### 策略 2: 用户组内均摊与软隔离

这个策略仅适用于用户组内部。我们引入了一个用户组内的用户配额配置表。该表定义了用户在 100% GPU 资源中的 “理想份额比例”。每个 “用户” 在列表中以 user_id 的形式出现，并且一个用户只能属于一个用户组。低于配额表上额定值的用户会比高于额定值的用户拥有更高的优先级获得被释放资源而进行更多的推理，直到双方使用量趋近于原始配额比例。此处调度只考虑请求队列中的用户，忽略没有出现在请求队列中的已配置用户。

以下图表展示了这种策略的典型示例。

![](https://github.com/InternLM/lmdeploy/assets/52888924/3e1d7135-6b11-4998-89a1-b72af6c962c3)

#### 策略3：混合机制

是指在一个系统中优先级+均摊/隔离同时开启。执行顺序是先用户组间优先级，再在组内做均摊/隔离实现。这里略去时序图描写。需要注意的是，用户组间的优先级可以压倒性覆盖组内的决策。例如，当低优先级内部的两个用户互相之间有请求顺序调度时，高优先级的请求一旦抵达，将会覆盖所有低优先级的分配逻辑而有限执行高优任务。

![](https://github.com/InternLM/lmdeploy/assets/52888924/e335f976-ff15-48db-b1ff-abf1c3327d6e)

需要注意的是，混合机制可能有其他方法，本文档只介绍了一种在我们场景下有效的方法。其他混合方法需要考虑到优先级和按比例共享明显是相互冲突的策略，因此没有简单的方法将它们混合在单一维度内工作。

### QoS 配置项模板

配置文件通过启动参数`--qos-config-path`指定，并由程序在启动时加载。

配置会和lmdeploy启动脚本等文件放置在一起。配置内容包含：

1. QoS的启用开关，设置为True时后续的QoS和用户相关配置才会生效，设置为False后续配置不会生效；

2. user_groups 是一个列表，包含了多种不同的组间优先级；

3. user_group_map 的映射配置，包含了用户组优先级，组内用户id以及每个用户组内用户的配额分配。

配置项模板如下：

```json
{
    "enable_user_qos": true,
    "user_groups": [
        "Platinum",
        "Gold",
        "Silver",
        "Bronze"
    ],
    "user_group_map": {
        "Platinum": [
            {
                "id": "user_id0",
                "quota_pct": 100
            },
            {
                "id": "default",
                "quota_pct": 0
            }
        ],
        "Gold": [
            {
                "id": "user_id1",
                "quota_pct": 50
            },
            {
                "id": "user_id2",
                "quota_pct": 50
            }
        ],
        "Silver": [
            {
                "id": "user_id3",
                "quota_pct": 5
            },
            {
                "id": "default",
                "quota_pct": 95
            }
        ],
        "Bronze": [
            {
                "id": "user_id4",
                "quota_pct": 30
            },
            {
                "id": "user_id5",
                "quota_pct": 30
            },
            {
                "id": "user_id6",
                "quota_pct": 40
            },
            {
                "id": "default",
                "quota_pct": 0
            }
        ]
    }
}
```

### 如何使用 LMDeploy-QoS 感知进行推理

我们提供以下代码链接，展示如何调用具有多租户策略感知的推理请求，在 HTTP Body 中，与 QoS 相关的参数如下：

/v1/chat/interactive_qos

```bash
curl -X POST http://localhost/v1/chat/interactive_qos \
  -H "Content-Type: application/json" \
  -d '{
  "prompt": "Hello,Hello",
  "session_id": -1,
  "interactive_mode": false,
  "stream": false,
  "stop": false,
  "request_output_len": 512,
  "top_p": 0.8,
  "top_k": 40,
  "temperature": 0.8,
  "repetition_penalty": 1,
  "ignore_eos": false,
  "user_id": "user_id0"
}'
```

/v1/chat/completions_qos

```bash
curl -X POST http://localhost/v1/chat/completions_qos \
  -H "Content-Type: application/json" \
  -d '{
  "model": "internlm-chat-7b",
  "messages": "Hello,Hello",
  "temperature": 0.7,
  "top_p": 1,
  "n": 1,
  "max_tokens": 512,
  "stop": false,
  "stream": false,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "repetition_penalty": 1,
  "session_id": -1,
  "ignore_eos": false,
  "user_id": "user_id0"
}'
```

/v1/completions_qos

```bash
curl -X POST http://localhost/v1/completions_qos \
  -H "Content-Type: application/json" \
  -d '{
  "model": "internlm-chat-7b",
  "prompt": "Hello,Hello",
  "suffix": "string",
  "temperature": 0.7,
  "n": 1,
  "max_tokens": 16,
  "stop": "string",
  "stream": false,
  "top_p": 1,
  "repetition_penalty": 1,
  "session_id": -1,
  "ignore_eos": false,
  "user_id": "user_id0"
}'
```

### 配置文件修改

配置文件模板路径为：`lmdeploy/server/qos_engine/qos_config.json.template`，可以根据实际需求添加需要配置的用户，设置正确的优先级以及quota值。

### 配置参数传入

启动api_server时，通过`--qos-config-path`，将配置文件及路径传入，示例如下：

```bash
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server internlm/internlm-chat-7b --server-port 8000 --qos-config-path lmdeploy/serve/qos_engine/qos_config.json.template
```

### 贡献者

[Eric](https://github.com/rhinouser0), [sallyjunjun](https://github.com/sallyjunjun), [sfireworks](https://github.com/sfireworks), [Dofgal](https://github.com/Dofgal), [shadow](https://github.com/awslshadowstar)

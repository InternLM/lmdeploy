## LMDeploy-QoS Introduce and Usage

### Background

With the rise of Large Language Model (LLM) and Artificial General Intelligence (AGI), numerous inference frameworks have emerged. These frameworks deliver scalable and high-performance services by serving online workloads with language models. However, these workloads often come from multiple user groups, exhibiting rapid changes in workload patterns within short periods. Many inference frameworks struggle to meet the demands of such multi-tenancy traffic patterns and fail to effectively shape user behaviors. Therefore, we believe that systematically considering these issues in LLM inference framework is both valuable and necessary.

### User Categorizations for Multi-tenancy Handling

LMDeploy-QoS is part of LMDeploy, offering a range of multi-tenancy functionalities. It requires users to tag their inference requests with appropriate user identifications (user_id in configuration or codebase). The system operates based on a dictionary-like configuration that serves as a multi-tenancy policy. In this configuration, users are mapped to different classes, known as "user groups", each configured with a ratio value. Our multi-tenancy strategy reads this configuration and schedules user inference requests according to class priority and the difference between the predefined ratio and real-time allocation ratio. Extensive testing shows that LMDeploy-QoS significantly enhances LLM serving reliability and GPU resource utilization for real-world large language model inference workloads.

We categorize LMDeploy users into four groups:

- Platinum
- Gold
- Silver
- Bronze

Based on our experiences in delivering LLM services, we can map the following four types of users to these user groups:

- Platinum: VIP or administrative users. Examples include service inspectors or product demo presenters who require uninterrupted online services. Their workloads are typically at a low frequency and require limited resources.

- Gold: Contracted business user groups requiring specific quantities of reliable services. For instance, Company A signs a contract with the LLM service provider to secure X requests/sec service capability with Z% availability for its employees at the cost of Y million dollars per year.

- Silver: The vast majority of users fall under this category. Most trial or monthly subscribed users are included in this group. They need a relatively small quantity of services, but their user experiences significantly affect the LLM service reputation.

- Bronze: Heavy users who pay minimal fees to LLM providers.

The above user group categorization is intended for guidance rather than as a recommendation for all LMDeploy users, as it may not be suitable for all LLM service providers. Users can develop their own method of categorizing users based on their observations of daily workloads.

Next, we will discuss how LMDeploy schedules requests based on these categorizations.

### Multi-tenancy Strategies

#### Strategy 1: prioritized scheduling between groups

This strategy works as simple as its title suggests.

User groups are introduced for this strategy, with users in each group to be specified. Recommended user groups are as follows:

- Platinum
- Gold
- Silver
- Bronze

The priority of each group decreases sequentially. Requests with higher priority are always given precedence for inference. Be noted that the scheduling is performed at the time of request reception, so lower-priority requests will not be withdrawn from the GPU if they are already under inference.

The below diagram shows how the prioritization works. As you can see, the platinum request is reprioritized and moved to the queue head.

![](https://github.com/InternLM/lmdeploy/assets/52888924/9d63f081-7168-4c74-8456-24f0a4b41649)

#### Strategy 2: proportionally rated scheduling with a pre-defined ratio within user group

This strategy works only within the user group. We introduce a within-group user quota configuration table. This table defines users' "ideal share ratio" with a sum value of 100% GPU resource. Each "user" appears in the list as a user_id, and a user can only belong to one user group. Requests from different users will be scheduled according to each user's "ideal share ratio". To be specific, users with their real-time usage ratio lower than their quota ratio will have priority over users whose real-time usage ratio is higher than their quota ratio. It is worth noting that the scheduling only considers users in the request queue, ignoring any absent users from the configuration table.

The below diagram shows a typical example of how this strategy works.

![](https://github.com/InternLM/lmdeploy/assets/52888924/3e1d7135-6b11-4998-89a1-b72af6c962c3)

#### Strategy 3: a combination strategy of 1 and 2

We can call it a hybrid strategy. The way we hybrid these 2 strategies is fairly simple: we adopt strategy 1 in between user groups, and adopt strategy 2 within a user group. So users belonging to different groups with different priorities will only obey strategy 1 to determine their privilege in resource allocation. That is, when both strategies are applied, the first strategy will overpower the second. When it comes to a situation that no cross-group requests are waiting for serving, the within-group strategy 2 comes into play.

Below is a diagram showing it.

![](https://github.com/InternLM/lmdeploy/assets/52888924/e335f976-ff15-48db-b1ff-abf1c3327d6e)

To be noted, there could be other ways of hybrid strategies 1 & 2, and this doc only introduces one method that works well in our scenario. Considering that prioritization and pro-rated sharing are obviously conflicting strategies, there is no easy way to mix them to work within a single dimension.

### A Sample QoS Configuration

The configuration will be specified by the `--qos-config-path` flag, and will be loaded by program upon startup.

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

### How to perform inference job with Lmdeploy-QoS aware

We provide the code link below to show how to call infer requests with multi-tenancy strategy awarded. What the qos related argument appears as in http body：

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

### File Configuration Modification

The template of the configuration file is located at: `lmdeploy/server/qos_engine/qos_config.json.template`. Add the necessary users based on actual requirements, ensure correct priority assignment, and set appropriate quota values.

### Passing Configuration Parameters

Upon starting the api_server, pass the configuration file and its path using the `--qos-config-path` flag. An example is illustrated below:

```bash
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server internlm/internlm-chat-7b --server-port 8000 --qos-config-path lmdeploy/serve/qos_engine/qos_config.json.template
```

### Contributor

[Eric](https://github.com/rhinouser0), [sallyjunjun](https://github.com/sallyjunjun), [sfireworks](https://github.com/sfireworks), [Dofgal](https://github.com/Dofgal), [shadow](https://github.com/awslshadowstar)

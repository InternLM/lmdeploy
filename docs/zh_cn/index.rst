欢迎来到 LMDeploy 的中文教程！
====================================

.. figure:: ./_static/image/lmdeploy-logo.svg
  :width: 50%
  :align: center
  :alt: LMDeploy
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LMDeploy 是一个高效且友好的 LLMs 模型部署工具箱，功能涵盖了量化、推理和服务
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

LMDeploy 工具箱提供以下核心功能：

- **高效的推理：** LMDeploy 开发了 Persistent Batch(即 Continuous Batch)，Blocked K/V Cache，动态拆分和融合，张量并行，高效的计算 kernel等重要特性。推理性能是 vLLM 的 1.8 倍

- **可靠的量化：** LMDeploy 支持权重量化和 k/v 量化。4bit 模型推理效率是 FP16 下的 2.4 倍。量化模型的可靠性已通过 OpenCompass 评测得到充分验证。

- **便捷的服务：** 通过请求分发服务，LMDeploy 支持多模型在多机、多卡上的推理服务。

- **有状态推理：** 通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

- **卓越的兼容性:**  LMDeploy 支持 `KV Cache 量化 <https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/kv_quant.html>`_, `AWQ <https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/w4a16.html>`_ 和 `Automatic Prefix Caching <https://lmdeploy.readthedocs.io/zh-cn/latest/inference/turbomind_config.html>`_ 同时使用。

中文文档
--------

.. _快速上手:
.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   get_started/installation.md
   get_started/get_started.md
   get_started/index.rst

.. _支持的模型:
.. toctree::
   :maxdepth: 1
   :caption: 模型列表

   supported_models/supported_models.md

.. _llm_部署:
.. toctree::
   :maxdepth: 1
   :caption: 大语言模型(LLMs)部署

   llm/pipeline.md
   llm/api_server.md
   llm/api_server_tools.md
   llm/api_server_lora.md
   llm/gradio.md
   llm/proxy_server.md

.. _vlm_部署:
.. toctree::
   :maxdepth: 1
   :caption: 视觉-语言模型(VLMs)部署

   multi_modal/vl_pipeline.md
   multi_modal/api_server_vl.md
   multi_modal/index.rst


.. _量化:
.. toctree::
   :maxdepth: 1
   :caption: 量化

   quantization/w4a16.md
   quantization/w8a8.md
   quantization/kv_quant.md

.. _测试基准:
.. toctree::
   :maxdepth: 1
   :caption: 测试基准

   benchmark/profile_generation.md
   benchmark/profile_throughput.md
   benchmark/profile_api_server.md
   benchmark/evaluate_with_opencompass.md

.. toctree::
   :maxdepth: 1
   :caption: 进阶指南

   inference/turbomind.md
   inference/pytorch.md
   advance/pytorch_new_model.md
   advance/long_context.md
   advance/chat_template.md
   advance/debug_turbomind.md
   advance/structed_output.md

.. toctree::
   :maxdepth: 1
   :caption: API 文档

   api/pipeline.rst

索引与表格
==================

* :ref:`genindex`
* :ref:`search`

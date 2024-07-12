Welcome to LMDeploy's tutorials!
====================================

.. figure:: ./_static/image/lmdeploy-logo.svg
  :width: 50%
  :align: center
  :alt: LMDeploy
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>LMDeploy is a toolkit for compressing, deploying, and serving LLM.
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/InternLM/lmdeploy/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

LMDeploy has the following core features:

* **Efficient Inference**: LMDeploy delivers up to 1.8x higher request throughput than vLLM, by introducing key features like persistent batch(a.k.a. continuous batching), blocked KV cache, dynamic split&fuse, tensor parallelism, high-performance CUDA kernels and so on.

* **Effective Quantization**: LMDeploy supports weight-only and k/v quantization, and the 4-bit inference performance is 2.4x higher than FP16. The quantization quality has been confirmed via OpenCompass evaluation.

* **Effortless Distribution Server**: Leveraging the request distribution service, LMDeploy facilitates an easy and efficient deployment of multi-model services across multiple machines and cards.

* **Interactive Inference Mode**: By caching the k/v of attention during multi-round dialogue processes, the engine remembers dialogue history, thus avoiding repetitive processing of historical sessions.

* **Excellent Compatibility**: LMDeploy supports `KV Cache Quant <https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html>`_, `AWQ <https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html>`_ and `Automatic Prefix Caching <https://lmdeploy.readthedocs.io/en/latest/inference/turbomind_config.html>`_ to be used simultaneously.

Documentation
-------------

.. _get_started:
.. toctree::
   :maxdepth: 2
   :caption: Get Started

   get_started.md

.. _build:
.. toctree::
   :maxdepth: 1
   :caption: Build

   build.md

.. _benchmark:
.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   benchmark/profile_generation.md
   benchmark/profile_throughput.md
   benchmark/profile_api_server.md
   benchmark/profile_triton_server.md
   benchmark/evaluate_with_opencompass.md

.. _supported_models:
.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models/supported_models.md

.. _inference:
.. toctree::
   :maxdepth: 1
   :caption: Inference

   inference/pipeline.md
   inference/vl_pipeline.md

.. _serving:
.. toctree::
   :maxdepth: 1
   :caption: Serving

   serving/api_server.md
   serving/api_server_vl.md
   serving/api_server_tools.md
   serving/gradio.md
   serving/proxy_server.md

.. _quantization:
.. toctree::
   :maxdepth: 1
   :caption: Quantization

   quantization/w4a16.md
   quantization/kv_quant.md
   quantization/w8a8.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Guide

   inference/turbomind.md
   inference/pytorch.md
   advance/pytorch_new_model.md
   advance/long_context.md
   advance/chat_template.md
   advance/debug_turbomind.md
   serving/qos.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/pipeline.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

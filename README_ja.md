<div align="center">
  <img src="docs/en/_static/image/lmdeploy-logo.svg" width="450"/>

[![PyPI](https://img.shields.io/pypi/v/lmdeploy)](https://pypi.org/project/lmdeploy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmdeploy)
[![license](https://img.shields.io/github/license/InternLM/lmdeploy.svg)](https://github.com/InternLM/lmdeploy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/lmdeploy)](https://github.com/InternLM/lmdeploy/issues)

[📘Documentation](https://lmdeploy.readthedocs.io/en/latest/) |
[🛠️Quick Start](https://lmdeploy.readthedocs.io/en/latest/get_started/get_started.html) |
[🤔Reporting Issues](https://github.com/InternLM/lmdeploy/issues/new/choose)

[English](README.md) | [简体中文](README_zh-CN.md) | 日本語

👋 join us on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/lmdeploy.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

</div>

______________________________________________________________________

## 最新ニュース 🎉

<details close>
<summary><b>2024</b></summary>

- \[2024/08\] 🔥🔥 LMDeployは[modelscope/swift](https://github.com/modelscope/swift)に統合され、VLMs推論のデフォルトアクセラレータとなりました
- \[2024/07\] 🎉🎉 Llama3.1 8B、70Bおよびそのツールコールをサポート
- \[2024/07\] [InternVL2](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e)全シリーズモデル、[InternLM-XComposer2.5](docs/en/multi_modal/xcomposer2d5.md)およびInternLM2.5の[ファンクションコール](docs/en/llm/api_server_tools.md)をサポート
- \[2024/06\] PyTorchエンジンはDeepSeek-V2およびいくつかのVLMs、例えばCogVLM2、Mini-InternVL、LlaVA-Nextをサポート
- \[2024/05\] 複数のGPUでVLMsをデプロイする際にビジョンモデルをバランスさせる
- \[2024/05\] InternVL v1.5、LLaVa、InternLMXComposer2などのVLMsで4ビットの重みのみの量子化と推論をサポート
- \[2024/04\] Llama3およびInternVL v1.1、v1.2、MiniGemini、InternLMXComposer2などのVLMモデルをサポート
- \[2024/04\] TurboMindはすべてのサポートされているデバイスでのオンラインint8/int4 KVキャッシュ量子化と推論を追加しました。詳細なガイドは[こちら](docs/en/quantization/kv_quant.md)を参照してください
- \[2024/04\] TurboMindの最新アップグレードによりGQAが強化され、[internlm2-20b](https://huggingface.co/internlm/internlm2-20b)モデルの推論が16+ RPSに達し、vLLMの約1.8倍の速さになりました
- \[2024/04\] Qwen1.5-MOEおよびdbrxをサポート
- \[2024/03\] DeepSeek-VLのオフライン推論パイプラインとサービングをサポート
- \[2024/03\] VLMのオフライン推論パイプラインとサービングをサポート
- \[2024/02\] Qwen 1.5、Gemma、Mistral、Mixtral、Deepseek-MOEなどをサポート
- \[2024/01\] [OpenAOE](https://github.com/InternLM/OpenAOE)が[LMDeployサービングサービス](./docs/en/llm/api_server.md)とシームレスに統合されました
- \[2024/01\] 複数モデル、複数マシン、複数カードの推論サービスをサポート。使用方法は[こちら](./docs/en/llm/proxy_server.md)を参照してください
- \[2024/01\] [PyTorch推論エンジン](./docs/en/inference/pytorch.md)をサポートし、完全にPythonで開発されており、開発者の障壁を下げ、新機能や技術の迅速な実験を可能にします

</details>

<details close>
<summary><b>2023</b></summary>

- \[2023/12\] Turbomindはマルチモーダル入力をサポート
- \[2023/11\] Turbomindはhfモデルの直接読み込みをサポート。詳細は[こちら](docs/en/inference/load_hf.md)をクリックしてください
- \[2023/11\] TurboMindの主要なアップグレード、包括的なPaged Attention、シーケンス長制限のない高速なアテンションカーネル、2倍速いKV8カーネル、Split-Kデコーディング（Flash Decoding）、およびsm_75のW4A16推論
- \[2023/09\] TurboMindはQwen-14Bをサポート
- \[2023/09\] TurboMindはInternLM-20Bをサポート
- \[2023/09\] TurboMindはCode Llamaのすべての機能をサポート：コード補完、インフィリング、チャット/インストラクト、Pythonスペシャリスト。デプロイメントガイドは[こちら](./docs/en/llm/codellama.md)をクリックしてください
- \[2023/09\] TurboMindはBaichuan2-7Bをサポート
- \[2023/08\] TurboMindはflash-attention2をサポート
- \[2023/08\] TurboMindはQwen-7B、動的NTK-RoPEスケーリング、動的logNスケーリングをサポート
- \[2023/08\] TurboMindはWindowsをサポート（tp=1）
- \[2023/08\] TurboMindは4ビット推論をサポートし、FP16の2.4倍の速さで、最速のオープンソース実装です。詳細な情報は[こちら](docs/en/quantization/w4a16.md)のガイドを確認してください
- \[2023/08\] LMDeployは[HuggingFace Hub](https://huggingface.co/lmdeploy)で提供され、すぐに使用できる4ビットモデルを提供します
- \[2023/08\] LMDeployは[AWQ](https://arxiv.org/abs/2306.00978)アルゴリズムを使用した4ビット量子化をサポート
- \[2023/07\] TurboMindはGQAを使用したLlama-2 70Bをサポート
- \[2023/07\] TurboMindはLlama-2 7B/13Bをサポート
- \[2023/07\] TurboMindはInternLMのテンソル並列推論をサポート

</details>

______________________________________________________________________

# 紹介

LMDeployは、[MMRazor](https://github.com/open-mmlab/mmrazor)および[MMDeploy](https://github.com/open-mmlab/mmdeploy)チームによって開発された、LLMの圧縮、デプロイ、およびサービングのためのツールキットです。以下の主要な機能を備えています：

- **効率的な推論**：LMDeployは、persistent batch（連続バッチ）、ブロック化されたKVキャッシュ、動的分割と融合、テンソル並列、高性能なCUDAカーネルなどの主要な機能を導入し、vLLMよりも最大1.8倍のリクエストスループットを提供します。

- **効果的な量子化**：LMDeployは、重みのみおよびk/vの量子化をサポートし、4ビットの推論性能はFP16の2.4倍です。量子化の品質はOpenCompassの評価を通じて確認されています。

- **簡単な分散サーバー**：リクエスト分散サービスを活用することで、LMDeployは複数のマシンおよびカードにわたるマルチモデルサービスのデプロイを容易にします。

- **優れた互換性**：LMDeployは、[KV Cache Quant](docs/en/quantization/kv_quant.md)、[AWQ](docs/en/quantization/w4a16.md)、および[Automatic Prefix Caching](docs/en/inference/turbomind_config.md)を同時に使用することをサポートします。

# パフォーマンス

LMDeploy TurboMindエンジンは卓越した推論能力を持ち、さまざまな規模のモデルで、vLLMの1.36〜1.85倍のリクエストを毎秒処理します。静的推論能力の面では、TurboMind 4ビットモデルの推論速度（out token/s）はFP16/BF16推論をはるかに上回ります。小さなバッチでは、2.4倍に向上します。

![v0 1 0-benchmark](https://github.com/InternLM/lmdeploy/assets/4560679/8e455cf1-a792-4fa8-91a2-75df96a2a5ba)

# サポートされているモデル

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>LLMs</b>
</td>
<td>
  <b>VLMs</b>
</td>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li>Llama (7B - 65B)</li>
  <li>Llama2 (7B - 70B)</li>
  <li>Llama3 (8B, 70B)</li>
  <li>Llama3.1 (8B, 70B)</li>
  <li>Llama3.2 (1B, 3B)</li>
  <li>InternLM (7B - 20B)</li>
  <li>InternLM2 (7B - 20B)</li>
  <li>InternLM3 (8B)</li>
  <li>InternLM2.5 (7B)</li>
  <li>Qwen (1.8B - 72B)</li>
  <li>Qwen1.5 (0.5B - 110B)</li>
  <li>Qwen1.5 - MoE (0.5B - 72B)</li>
  <li>Qwen2 (0.5B - 72B)</li>
  <li>Qwen2-MoE (57BA14B)</li>
  <li>Qwen2.5 (0.5B - 32B)</li>
  <li>Qwen3, Qwen3-MoE</li>
  <li>Qwen3-Next(80B)</li>
  <li>Baichuan (7B)</li>
  <li>Baichuan2 (7B-13B)</li>
  <li>Code Llama (7B - 34B)</li>
  <li>ChatGLM2 (6B)</li>
  <li>GLM-4 (9B)</li>
  <li>GLM-4-0414 (9B, 32B)</li>
  <li>CodeGeeX4 (9B)</li>
  <li>YI (6B-34B)</li>
  <li>Mistral (7B)</li>
  <li>DeepSeek-MoE (16B)</li>
  <li>DeepSeek-V2 (16B, 236B)</li>
  <li>DeepSeek-V2.5 (236B)</li>
  <li>Mixtral (8x7B, 8x22B)</li>
  <li>Gemma (2B - 7B)</li>
  <li>StarCoder2 (3B - 15B)</li>
  <li>Phi-3-mini (3.8B)</li>
  <li>Phi-3.5-mini (3.8B)</li>
  <li>Phi-3.5-MoE (16x3.8B)</li>
  <li>Phi-4-mini (3.8B)</li>
  <li>MiniCPM3 (4B)</li>
  <li>SDAR (1.7B-30B)</li>
</ul>
</td>
<td>
<ul>
  <li>LLaVA(1.5,1.6) (7B-34B)</li>
  <li>InternLM-XComposer2 (7B, 4khd-7B)</li>
  <li>InternLM-XComposer2.5 (7B)</li>
  <li>Qwen-VL (7B)</li>
  <li>Qwen2-VL (2B, 7B, 72B)</li>
  <li>Qwen2.5-VL (3B, 7B, 72B)</li>
  <li>Qwen3-VL (2B - 235B)</li>
  <li>DeepSeek-VL (7B)</li>
  <li>DeepSeek-VL2 (3B, 16B, 27B)</li>
  <li>InternVL-Chat (v1.1-v1.5)</li>
  <li>InternVL2 (1B-76B)</li>
  <li>InternVL2.5(MPO) (1B-78B)</li>
  <li>InternVL3 (1B-78B)</li>
  <li>InternVL3.5 (1B-241BA28B)</li>
  <li>Intern-S1 (241B)</li>
  <li>Intern-S1-mini (8.3B)</li>
  <li>Mono-InternVL (2B)</li>
  <li>ChemVLM (8B-26B)</li>
  <li>CogVLM-Chat (17B)</li>
  <li>CogVLM2-Chat (19B)</li>
  <li>MiniCPM-Llama3-V-2_5</li>
  <li>MiniCPM-V-2_6</li>
  <li>Phi-3-vision (4.2B)</li>
  <li>Phi-3.5-vision (4.2B)</li>
  <li>GLM-4V (9B)</li>
  <li>GLM-4.1V-Thinking (9B)</li>
  <li>Llama3.2-vision (11B, 90B)</li>
  <li>Molmo (7B-D,72B)</li>
  <li>Gemma3 (1B - 27B)</li>
  <li>Llama4 (Scout, Maverick)</li>
</ul>
</td>
</tr>
</tbody>
</table>

LMDeployは、[TurboMind](./docs/en/inference/turbomind.md)および[PyTorch](./docs/en/inference/pytorch.md)の2つの推論エンジンを開発しました。それぞれ異なる焦点を持っています。前者は推論性能の究極の最適化を目指し、後者は完全にPythonで開発されており、開発者の障壁を下げることを目指しています。

サポートされているモデルの種類や推論データタイプに違いがあります。各エンジンの能力については[この表](./docs/en/supported_models/supported_models.md)を参照し、実際のニーズに最適なものを選択してください。

# クイックスタート [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)

## インストール

クリーンなconda環境（Python 3.9 - 3.12）でlmdeployをインストールすることをお勧めします。

```shell
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
pip install lmdeploy
```

v0.3.0から、デフォルトの事前構築済みパッケージはCUDA 12でコンパイルされています。
CUDA 11+プラットフォームでのインストールに関する情報、またはソースからのビルド手順については、[インストールガイドを](docs/en/get_started/installation.md)参照してください。

## オフラインバッチ推論

```python
import lmdeploy
with lmdeploy.pipeline("internlm/internlm3-8b-instruct") as pipe:
    response = pipe(["Hi, pls intro yourself", "Shanghai is"])
    print(response)
```

> \[!NOTE\]
> デフォルトでは、LMDeployはHuggingFaceからモデルをダウンロードします。ModelScopeからモデルを使用する場合は、`pip install modelscope`コマンドでModelScopeをインストールし、環境変数を設定してください：
>
> `export LMDEPLOY_USE_MODELSCOPE=True`
>
> openMind Hubからモデルを使用する場合は、`pip install openmind_hub`コマンドでopenMind Hubをインストールし、環境変数を設定してください：
>
> `export LMDEPLOY_USE_OPENMIND_HUB=True`

推論パイプラインに関する詳細情報は[こちら](./docs/en/llm/pipeline.md)を参照してください。

# チュートリアル

LMDeployの基本的な使用方法については、[getting_started](docs/en/get_started/get_started.md)セクションを参照してください。

詳細なユーザーガイドと高度なガイドについては、[チュートリアル](https://lmdeploy.readthedocs.io/en/latest/)を参照してください：

- ユーザーガイド
  - [LLM推論パイプライン](./docs/en/llm/pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95#scrollTo=YALmXnwCG1pQ)
  - [VLM推論パイプライン](./docs/en/multi_modal/vl_pipeline.md) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF?usp=sharing)
  - [LLMサービング](docs/en/llm/api_server.md)
  - [VLMサービング](docs/en/multi_modal/api_server_vl.md)
  - [量子化](docs/en/quantization)
- 高度なガイド
  - [推論エンジン - TurboMind](docs/en/inference/turbomind.md)
  - [推論エンジン - PyTorch](docs/en/inference/pytorch.md)
  - [カスタムチャットテンプレート](docs/en/advance/chat_template.md)
  - [新しいモデルの追加](docs/en/advance/pytorch_new_model.md)
  - gemmチューニング
  - [長文推論](docs/en/advance/long_context.md)
  - [マルチモデル推論サービス](docs/en/llm/proxy_server.md)

# サードパーティプロジェクト

- LMDeployを使用してNVIDIA JetsonプラットフォームでLLMをオフラインでデプロイ：[LMDeploy-Jetson](https://github.com/BestAnHongjun/LMDeploy-Jetson)
- LMDeployとBentoMLを使用してLLMをデプロイするためのサンプルプロジェクト：[BentoLMDeploy](https://github.com/bentoml/BentoLMDeploy)

# 貢献

LMDeployへのすべての貢献に感謝します。貢献ガイドラインについては、[CONTRIBUTING.md](.github/CONTRIBUTING.md)を参照してください。

# 謝辞

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

# 引用

```bibtex
@misc{2023lmdeploy,
    title={LMDeploy: A Toolkit for Compressing, Deploying, and Serving LLM},
    author={LMDeploy Contributors},
    howpublished = {\url{https://github.com/InternLM/lmdeploy}},
    year={2023}
}
```

# ライセンス

このプロジェクトは[Apache 2.0ライセンス](LICENSE)の下でリリースされています。

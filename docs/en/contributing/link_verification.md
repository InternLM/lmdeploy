# External Link Verification

This document tracks the verification status of external links in LMDeploy documentation.

## Last Verified: 2026-06-15

All external links in README.md have been verified and are accessible.

## Verified Links

### Badge Links
- ✅ PyPI: https://pypi.org/project/lmdeploy
- ✅ License: https://github.com/InternLM/lmdeploy/tree/main/LICENSE
- ✅ GitHub Issues: https://github.com/InternLM/lmdeploy/issues

### Documentation Links
- ✅ Main Docs: https://lmdeploy.readthedocs.io/en/latest/
- ✅ Quick Start: https://lmdeploy.readthedocs.io/en/latest/get_started/get_started.html
- ✅ Issue Reporter: https://github.com/InternLM/lmdeploy/issues/new/choose

### Community Links
- ✅ WeChat: https://cdn.vansin.top/internlm/lmdeploy.jpg
- ✅ Twitter: https://twitter.com/intern_lm
- ✅ Discord: https://discord.gg/xa29JuW87d

### Colab Notebooks
- ✅ LLM Pipeline: https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95
- ✅ VLM Pipeline: https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF

### Model References
- ✅ Qwen3.5 Collection: https://huggingface.co/collections/Qwen/qwen35

### Integration Partners
- ✅ DLSlime: https://github.com/DeepLink-org/DLSlime
- ✅ Mooncake: https://github.com/kvcache-ai/Mooncake
- ✅ ModelScope Swift: https://github.com/modelscope/swift
- ✅ OpenAOE: https://github.com/InternLM/OpenAOE

### Third-party Projects
- ✅ LMDeploy-Jetson: https://github.com/BestAnHongjun/LMDeploy-Jetson
- ✅ BentoLMDeploy: https://github.com/bentoml/BentoLMDeploy

### Acknowledgements
- ✅ FasterTransformer: https://github.com/NVIDIA/FasterTransformer
- ✅ llm-awq: https://github.com/mit-han-lab/llm-awq
- ✅ vLLM: https://github.com/vllm-project/vllm
- ✅ DeepSpeed-MII: https://github.com/microsoft/DeepSpeed-MII

## Verification Method

Links were verified using Python's `urllib.request` with a 5-second timeout:

```python
import urllib.request

urls = [
    "https://colab.research.google.com/drive/1Dh-YlSwg78ZO3AlleO441NF_QP2shs95",
    "https://colab.research.google.com/drive/1nKLfnPeDA3p-FMNw2NhI-KOpk7-nlNjF",
    # ... other URLs
]

for url in urls:
    try:
        response = urllib.request.urlopen(url, timeout=5)
        status = response.getcode()
        print(f"{url}: {'OK' if status == 200 else f'FAIL ({status})'}")
    except Exception as e:
        print(f"{url}: ERROR - {e}")
```

## Automated Link Checking (Future)

Consider adding automated link checking to CI/CD:

```yaml
# .github/workflows/link-check.yml
name: Link Check

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  link-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check links
        uses: lycheeverse/lychee-action@v1
        with:
          args: README.md docs/**/*.md
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Maintenance Schedule

- **Monthly**: Verify critical links (Colab, documentation)
- **Quarterly**: Full link audit of all markdown files
- **Before releases**: Verify all README links

## Reporting Broken Links

If you find a broken link, please:
1. Open an issue at https://github.com/InternLM/lmdeploy/issues
2. Include the broken URL and where it was found
3. Suggest a replacement if available

Or submit a PR directly fixing the link.

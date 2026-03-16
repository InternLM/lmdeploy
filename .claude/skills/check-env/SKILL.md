---
name: check-env
description: Check if the LMDeploy dev environment is properly set up.
---

# LMDeploy Environment Setup

## Quick Start

```bash
conda env list                    # Find env name (usually 'dev')
conda activate dev                # Replace 'dev' with actual name
python -c "import lmdeploy; print(lmdeploy.__file__)"  # Should point into repo dir
which python                      # Confirm conda env path
```

## Verify

- `conda activate` succeeds without error
- `python -c "import lmdeploy; print(lmdeploy.__file__)"` points into the repo dir (editable install)
- `which python` shows conda env path (not `/usr/bin/python`)

## Troubleshooting

| Problem              | Solution                                     |
| -------------------- | -------------------------------------------- |
| `conda: not found`   | `source ~/miniconda3/etc/profile.d/conda.sh` |
| `lmdeploy` not found | `pip install -e .` from repo root            |
| Wrong Python         | `conda deactivate && conda activate dev`     |

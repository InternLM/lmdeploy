---
name: check-env
description: Check if the LMDeploy dev environment is properly set up.
---

# Check LMDeploy Dev Environment

## 1. Find and activate the conda env

```bash
conda env list                        # starred = currently active
conda activate <env-name>             # pick the right env for this project
```

## 2. Verify editable install

```bash
python -c "import lmdeploy; print(lmdeploy.__file__)"
# Must point into the repo dir, e.g. /path/to/lmdeploy_vl/lmdeploy/__init__.py
```

If it doesn't:

```bash
pip install -e .                      # run from repo root
```

## 3. Confirm python and CUDA

```bash
which python                          # must show conda env path, not /usr/bin/python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

## Troubleshooting

| Problem              | Fix                                             |
| -------------------- | ----------------------------------------------- |
| `conda: not found`   | `source ~/miniconda3/etc/profile.d/conda.sh`    |
| Wrong Python         | `conda deactivate && conda activate <env-name>` |
| `lmdeploy` not found | `pip install -e .` from repo root               |

---
name: submit-pr
description: Submit a GitHub pull request for LMDeploy.
---

# Submit a PR for LMDeploy

## 1. Create branch (off main)

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>   # e.g. feat/qwen3-omni
```

## 2. Lint

```bash
pre-commit run --all-files
```

Fix any reported issues before staging.

## 3. Stage

```bash
git add lmdeploy/path/to/changed_file.py     # specific files only, never git add .
git status                                   # verify staged set
```

## 4. Commit

```bash
git commit -m "feat: add Qwen3-Omni support"
# Conventional prefixes: feat | fix | refactor | docs | test | chore
```

## 5. Push

```bash
git push -u origin <branch>
```

## 6. Create PR

```bash
gh pr create --title "<type>: <short description>" --body "$(cat <<'EOF'
## Summary
- <bullet 1>
- <bullet 2>

## Test plan
- [ ] `pre-commit run --all-files` passes
- [ ] unit tests pass: `pytest tests/test_lmdeploy/`
- [ ] manual smoke test with pipeline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Checklist

| Check                            | Command                               |
| -------------------------------- | ------------------------------------- |
| Lint clean                       | `pre-commit run --all-files`          |
| Unit tests pass                  | `pytest tests/test_lmdeploy/`         |
| VL tests pass (if VLM change)    | `pytest tests/test_lmdeploy/test_vl/` |
| Branch off `main`                | `git log --oneline main..HEAD`        |
| No secrets or debug files staged | `git diff --cached --name-only`       |
| PR targets `main`                | `gh pr view --json baseRefName`       |

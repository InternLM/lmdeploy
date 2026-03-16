---
name: resolve-review
description: Fetch and resolve PR review comments, then push fixes.
---

# Resolve PR Review Comments

## 1. Fetch comments

```bash
gh api repos/InternLM/lmdeploy/pulls/<PR>/comments \
  | python3 -c "
import json, sys
for c in json.load(sys.stdin):
    print(f'[{c[\"path\"]}:{c.get(\"line\",\"?\")}]')
    print(c['body'])
    print()
"
```

## 2. Fix each issue

Read the flagged file, understand the comment, edit the file.

## 3. Lint

```bash
pre-commit run --all-files
```

## 4. Stage & commit

```bash
git add <fixed files>
git commit -m "fix: address PR review comments"
```

## 5. Push

```bash
git push
```

## Checklist

| Step                 | Done? |
| -------------------- | ----- |
| All comments read    | \[ \] |
| Files fixed          | \[ \] |
| `pre-commit` passes  | \[ \] |
| Committed and pushed | \[ \] |

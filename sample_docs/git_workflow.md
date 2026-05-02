# Git Workflow and Team Collaboration

This guide covers Git fundamentals, branching models, merge strategies, automation hooks, and how Git fits into CI/CD pipelines for application delivery.

## Git basics

Git tracks snapshots of a repository as commits. Each commit points to a tree of files and has a parent commit (except the root).

```bash
git init
git status
git add .
git commit -m "Initial commit"
```

Remote workflows typically use `clone`, `fetch`, `pull`, and `push`:

```bash
git clone https://github.com/org/repo.git
cd repo
git checkout -b feature/login
# ... edit files ...
git add -A
git commit -m "Add login form"
git push -u origin feature/login
```

Useful inspection commands:

```bash
git log --oneline --graph --decorate -20
git diff
git show HEAD
git blame path/to/file.py
```

## Branching strategies

### GitFlow

GitFlow uses long-lived `main` (production), `develop` (integration), feature branches, release branches, and hotfix branches. It suits scheduled releases and teams that need strict separation between stabilization and development.

```bash
git checkout develop
git checkout -b feature/cart
# merge back via PR into develop
git checkout -b release/1.2.0 develop
# finalize version, merge to main and develop
```

### Trunk-based development

Trunk-based development favors short-lived branches merged frequently into a single mainline (`main`). Feature flags hide incomplete work. This reduces merge pain and aligns with continuous delivery.

```bash
git checkout main
git pull
git checkout -b short-task-123
# small change, fast review, merge same day
```

### Choosing a model

| Model | Best when |
|-------|-----------|
| GitFlow | Versioned releases, QA gates, multiple supported versions |
| Trunk-based | High deployment frequency, strong automation, small batches |

## Merge vs rebase

**Merge** preserves branch history with a merge commit:

```bash
git checkout main
git merge feature/foo
```

**Rebase** replays commits on top of another branch for a linear history:

```bash
git checkout feature/foo
git fetch origin
git rebase origin/main
```

Rebase before opening a PR to simplify review; avoid rebasing commits already pushed to shared branches unless the team agrees.

## Conflict resolution

Conflicts occur when the same lines diverged. Git marks conflict regions:

```text
<<<<<<< HEAD
our change
=======
their change
>>>>>>> branch-name
```

Resolve by editing the file, then:

```bash
git add resolved_file.py
git rebase --continue   # or git commit after merge
```

For binary conflicts, choose ours/theirs explicitly:

```bash
git checkout --ours path/to/asset.png
git add path/to/asset.png
```

## Git hooks

Hooks are scripts in `.git/hooks/` (or managed via frameworks like `pre-commit`).

Example pre-commit (syntax check):

```bash
#!/bin/sh
python -m compileall -q .
```

Client-side hooks run locally; server-side hooks enforce policy on push (e.g., `pre-receive` rejecting unsigned commits).

## CI/CD integration

Typical pipeline stages:

1. **Lint & test** on every push/PR.
2. **Build** artifacts or container images.
3. **Deploy** to staging automatically; production with approval or tags.

```yaml
# Example GitHub Actions outline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest
```

Protect `main` with required status checks and code review rules.

## Conventional commits

Conventional Commits standardize messages for changelog generation and semantic versioning:

```text
feat(api): add user search endpoint
fix(auth): handle expired refresh tokens
docs: clarify deployment steps
chore: bump pytest to 8.x
```

Tools can map `feat` → minor version and `fix` → patch version.

## Monorepo considerations

Monorepos store multiple packages in one repository. Benefits: atomic cross-project changes, unified tooling. Challenges: large clones, longer CI unless scoped.

Mitigations:

- Path filters in CI (`paths: ['services/api/**']`).
- Workspace tools (`npm`, `pnpm`, `uv`, `Bazel`).
- CODEOWNERS for directory-level review routing.

## Practical tips

- Keep commits small and focused; write messages that explain *why*.
- Prefer `git pull --rebase` on feature branches to reduce noise.
- Tag releases (`v1.4.0`) and record deployment metadata.
- Never commit secrets; use environment variables and secret managers.

This document is intended as internal reference material for engineering onboarding and release hygiene.

## Signed commits and verification

GPG or SSH-signed commits prove authorship. Configure signing and teach reviewers to verify tags on releases.

```bash
git config --global user.signingkey YOURKEY
git config --global commit.gpgsign true
```

## Bisect for regressions

Binary search history to locate a bad commit:

```bash
git bisect start
git bisect bad                 # current broken
git bisect good v1.3.0         # last known good
# Git checks out midpoints; mark good/bad until culprit found
git bisect reset
```

## Submodules and subtrees

Git submodules pin external repositories at specific commits—useful for shared libraries, but they complicate clones and CI. Subtrees merge external history into your repo; choose based on update frequency and team familiarity.

## Release tagging and changelogs

Automate changelog generation from conventional commits or labeled PRs. Tag annotated releases:

```bash
git tag -a v2.0.0 -m "Release 2.0.0"
git push origin v2.0.0
```

## Large file handling

Avoid committing binaries. Use Git LFS when large assets are unavoidable, or store assets in object storage with version metadata in Git.

## Review etiquette

Keep PRs small, describe motivation and risk, link issues, and prefer constructive review comments. Protect `main` with required reviews and passing checks.

These workflows reinforce predictable delivery and auditability across teams.

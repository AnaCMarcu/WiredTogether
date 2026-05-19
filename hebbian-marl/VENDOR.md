# Vendoring record

The `epymarl/` directory is a **vendored copy** of upstream EPyMARL, not a submodule.

## Upstream

- **Repository:** https://github.com/uoe-agents/epymarl
- **Branch:** main
- **Commit:** `cbc38c09588064eab978501d0f12c2cf58fa7fc2`
- **Date:** 2024-09-24
- **Commit message:** Update README.md

## Why vendored

We add new files alongside the existing source (runner subclass, learner subclass, comm wrapper, hebbian_module). A submodule would prevent that. Vendored source lets us treat upstream as part of our codebase while keeping the upstream version pinned via this record.

## To re-sync with upstream

```bash
# In a scratch directory:
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl
git log --oneline -1                # confirm new commit
# Then manually merge changes into hebbian-marl/epymarl/, updating this VENDOR.md
```

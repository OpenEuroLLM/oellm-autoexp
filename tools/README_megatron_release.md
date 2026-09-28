# Standalone Megatron version packaging and publication

`megatron_release.py` packages an already committed, clean, tested Megatron revision into an isolated autoexp release. Python 3.10+ and Git are sufficient; publication to GitHub uses an existing authenticated `gh`. No package installation, training imports, network services or GPUs are needed for local packaging. Run in an activated project environment and a persistent terminal for long network transfers. Dry-run is the default for both preparation and publication.

Start with clean committed source checkouts. Commit your validation document in autoexp first; the release records its exact SHA-256. Do not publish test credentials, datasets or generated training output. The helper refuses dirty/untracked source, protected index flags, differing submodule URLs, nested unhandled Megatron submodules and existing output/version paths. It clones committed files/history independently, leaving original refs, indices, worktrees, ignored build products and live jobs alone.

```bash
python tools/megatron_release.py prepare \
  --autoexp /path/to/clean-autoexp \
  --megatron /path/to/clean-tested-Megatron-LM \
  --output /path/to/new-release-package \
  --name oellm-32b-b1-reborn-20260925 \
  --autoexp-branch prod/oellm_32b_dense_b1_reborn \
  --megatron-branch oellm/v0.19-b1-reborn \
  --evidence docs/megatron_versions/b1_reborn_20260925.md
# Review the identities, destinations and evidence; repeat with --execute.
```

For a later release choose a new immutable `--name`, the intended branches and a new reviewed evidence document. Source commits and the complete Megatron tree are preserved. The new autoexp commit changes only its Megatron gitlink and `versions/megatron/NAME.json`; the latter contains no local source/output paths. Local `plan.json` and `release.json` retain staging paths and must stay outside Git. Optional `--assisted-by MODEL` appends an attribution to the generated commit. Repeating preparation refuses an existing directory rather than overwriting evidence; verify or publish that package instead.

```bash
python tools/megatron_release.py verify --package /path/to/new-release-package
gh auth status --hostname github.com
gh api repos/OpenEuroLLM/NVIDIA-Megatron-LM --jq .permissions.push
gh api repos/OpenEuroLLM/oellm-autoexp --jq .permissions.push
python tools/megatron_release.py publish --package /path/to/new-release-package
# Review the remote refs, then publish:
python tools/megatron_release.py publish --package /path/to/new-release-package --execute
python tools/megatron_release.py verify-remote --package /path/to/new-release-package
```

Publication uses command-scoped `credential.helper=!gh auth git-credential`; credentials are never extracted or persisted by the tool. It first publishes and independently checks the Megatron branch/tag, then the autoexp branch/tag. Each repository push is atomic. The pair cannot be one cross-repository transaction: if autoexp publication fails, keep the package, resolve the reported conflict and rerun publication. A reachable Megatron release may exist before autoexp, but autoexp is never deliberately published before its dependency. Existing identical refs are safe to retry; changed tags and non-fast-forward branches are refused, without force pushes. Re-prepare under a new version if another owner's branch has moved incompatibly; never erase their history.

`verify-remote` performs a fresh checkout from the autoexp remote tag, initializes only Megatron from `.gitmodules`, checks the actual dependency commit/tree and the evidence hash, and writes `remote-verification.json`. This is the collaborator availability check; a local commit or successful push alone is not sufficient. It downloads source history, so budget local temporary disk space. It does not fetch Titan, containers, caches or model weights. Keep the release package and receipts for reproducibility.

CPU tests (real temporary Git repositories and bare remotes, no network):

```bash
python -m unittest discover -s tests -p test_megatron_release.py -v
```

This routine packages source. It does not choose an experiment's LR/decay/precision settings, alter checkpoint optimizer state, start training or replace production monitors. Use the version-specific guide for tested settings and known scope.

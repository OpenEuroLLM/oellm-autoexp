#!/bin/bash
# Allow --cross-entropy-fusion-impl te in Megatron-LM oellm/v0.19 (7f9b593).
# Upstream disabled TE fused cross-entropy (NVIDIA/Megatron-LM#5115) and re-enabled it
# on main (#7339); that change is not in oellm/v0.19 yet. TE CE is what lets MBS 4 fit
# on 64 GB GPUs with a 256k vocab (no extra fp32 copy of the logits).
# - NeMo 26.08 (TE 2.16): pre-fix TE kernel -> benchmark use only.
# - NeMo 26.08.01, /opt/venv (TE 2.18): fixed kernel.
# Remove this script once oellm/v0.19 includes NVIDIA/Megatron-LM#7339.
set -e
M=submodules/Megatron-LM
if git -C "$M" apply --reverse --check - >/dev/null 2>&1 <<'EOF_CHECK'
diff --git a/megatron/training/arguments.py b/megatron/training/arguments.py
index 7c64ea580..fab30a939 100644
--- a/megatron/training/arguments.py
+++ b/megatron/training/arguments.py
@@ -1776,12 +1776,12 @@ def validate_args(args, defaults={}):
         assert args.fim_spm_rate, "--fim-spm-rate should be specified."
         assert all(token is not None for token in extra_tokens), "FIM extra tokens should be specified."

-    assert not (
-        args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == 'te'
-    ), (
-        "Transformer Engine cross entropy loss fusion is disabled due to stability issues. "
-        "Use --cross-entropy-fusion-impl native, or omit --cross-entropy-loss-fusion."
-    )
+#     assert not (
+#         args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == 'te'
+#     ), (
+#         "Transformer Engine cross entropy loss fusion is disabled due to stability issues. "
+#         "Use --cross-entropy-fusion-impl native, or omit --cross-entropy-loss-fusion."
+#     )

     # Deterministic mode — env vars + config overrides + torch global state.
     # Implementation lives in ``megatron/training/determinism.py`` so the
EOF_CHECK
then
    echo "TE cross-entropy patch already applied, nothing to do."
    exit 0
fi
git -C "$M" apply <<'EOF_PATCH'
diff --git a/megatron/training/arguments.py b/megatron/training/arguments.py
index 7c64ea580..fab30a939 100644
--- a/megatron/training/arguments.py
+++ b/megatron/training/arguments.py
@@ -1776,12 +1776,12 @@ def validate_args(args, defaults={}):
         assert args.fim_spm_rate, "--fim-spm-rate should be specified."
         assert all(token is not None for token in extra_tokens), "FIM extra tokens should be specified."

-    assert not (
-        args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == 'te'
-    ), (
-        "Transformer Engine cross entropy loss fusion is disabled due to stability issues. "
-        "Use --cross-entropy-fusion-impl native, or omit --cross-entropy-loss-fusion."
-    )
+#     assert not (
+#         args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == 'te'
+#     ), (
+#         "Transformer Engine cross entropy loss fusion is disabled due to stability issues. "
+#         "Use --cross-entropy-fusion-impl native, or omit --cross-entropy-loss-fusion."
+#     )

     # Deterministic mode — env vars + config overrides + torch global state.
     # Implementation lives in ``megatron/training/determinism.py`` so the
EOF_PATCH
echo "TE cross-entropy patch applied to $M."

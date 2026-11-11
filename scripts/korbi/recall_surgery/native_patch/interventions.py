"""Architecture interventions on the 1.7B ladder hybrids (native signed_kda Megatron codebase).

Recall surgery (docs/recall_surgery_ambitious.md): the ladder's `skda-shipped-hybrid-lowrank` stack is
CKDA in 3 of every 4 layers and fla's gated NoPE attention (MHA, 32 x 64 at 1.7B) in every 4th. These
interventions change that stack *as surgery* on its stable-phase checkpoint: every CKDA layer keeps its
weights and its complex eigenvalues, the model at step 0 is the original one, and only what an arm names
is added or switched.

Selected by the environment (read once, at spec-build time):

    SKDA_INTERVENTION="<item>[+<item>...]"

items:
    add_attn@<layers>          a parallel fla attention branch (the hybrid's own class and kwargs) next to
                               the CKDA mixer of the listed 1-based layers: out = ckda(x) + branch(x); the
                               branch's o_proj is zero at insertion (surgery), so the model is unchanged
    add_swa@<window>@<layers>  the same with a sliding window of <window> tokens (local copy path)
    attn_rope                  RoPE on the stack's EXISTING attention layers (theta 10000); their weights
                               are kept (they were trained NoPE)
    attn_kv@<n>                the stack's EXISTING attention layers rebuilt as GQA with <n> KV heads (the
                               query heads, o_proj and g_proj unchanged); surgery.py writes k_proj / v_proj
                               as the per-group MEANS of the original heads (GQA uptraining, Ainslie et al.
                               2023). Query head h reads KV head h // (num_heads / n), as flash-attn and
                               SDPA(enable_gqa) group them. Changes the function (the KV-capacity reverse
                               test, docs/recall_surgery_ambitious.md section 3.7)
    add_rope                   RoPE on the added branches
    <layers>: comma list of 1-based layer numbers, or "lin" = every CKDA layer, or "lin_late" = the CKDA
    layers of the second half (keeps the first layers untouched), or "post_attn" = the layer right after
    each attention layer

The branch tensors come from scripts/korbi/recall_surgery/native_patch/surgery.py (written into a copy of
the checkpoint), so the strict loader of the native codebase needs no change.
"""

from __future__ import annotations

import copy
import os

from megatron.core.transformer.spec_utils import ModuleSpec

from .mixer import FLAMixer, arm_kwargs


def parse(spec: str | None = None) -> dict:
    """{"add": [(layers, window|None)], "attn_rope": bool, "add_rope": bool, "attn_kv": int|None} from
    SKDA_INTERVENTION."""
    spec = (spec if spec is not None else os.environ.get("SKDA_INTERVENTION", "")).strip()
    out = {"add": [], "attn_rope": False, "add_rope": False, "attn_kv": None}
    if not spec or spec in ("none", "ctrl"):
        return out
    for item in spec.split("+"):
        parts = item.split("@")
        name = parts[0]
        if name == "add_attn":
            out["add"].append((parts[1], None))
        elif name == "add_swa":
            out["add"].append((parts[2], int(parts[1])))
        elif name == "attn_rope":
            out["attn_rope"] = True
        elif name == "add_rope":
            out["add_rope"] = True
        elif name == "attn_kv":
            out["attn_kv"] = int(parts[1])
            if out["attn_kv"] < 1:
                raise ValueError(f"SKDA_INTERVENTION item {item!r}: need >= 1 KV heads")
        else:
            raise ValueError(f"SKDA_INTERVENTION item {item!r}: unknown")
    return out


def layer_set(token: str, num_layers: int, every: int | None) -> list[int]:
    """1-based layer numbers of a layers token. ``every``: the stack's attention period (None: a pure
    linear stack, every layer is CKDA). Tokens: "lin" (every CKDA layer), "lin_late" (the CKDA layers of
    the second half), "every<k>" (every k-th layer, counting from 1, as the hybrids place attention),
    or a comma list."""
    lin = [i for i in range(1, num_layers + 1) if not every or i % every != 0]
    if token == "lin":
        return lin
    if token == "lin_late":
        return [i for i in lin if i > num_layers // 2]
    if token == "post_attn":
        # the layer right after each attention layer (a second attention step behind the first)
        if not every:
            raise ValueError("post_attn needs a hybrid stack (attention every k layers)")
        return [i for i in range(2, num_layers + 1) if (i - 1) % every == 0]
    if token.startswith("every"):
        k = int(token[len("every"):])
        return [i for i in range(1, num_layers + 1) if i % k == 0]
    return sorted(int(x) for x in token.split(","))


def branch_layers(iv: dict, num_layers: int, every: int | None) -> dict[int, int | None]:
    """{1-based layer: window or None} of every added branch."""
    out = {}
    for token, window in iv["add"]:
        for layer in layer_set(token, num_layers, every):
            if every and layer % every == 0:
                raise ValueError(f"layer {layer} is an attention layer already; add branches to CKDA layers")
            out[layer] = window
    return out


def branch_kwargs(hybrid_arm: str, config, layer_number: int, window: int | None, rope: bool,
                  num_kv_heads: int | None = None):
    """(class, kwargs) of a branch: the hybrid's attention class and kwargs (+ window / rope / KV heads)."""
    head_dim = config.kv_channels or (config.hidden_size // config.num_attention_heads)
    cls, kw = arm_kwargs(hybrid_arm, config.hidden_size, config.num_attention_heads, head_dim,
                         layer_number - 1, config.num_layers, config.layernorm_epsilon, role="attn")
    kw = dict(kw)
    if window:
        kw["window_size"] = int(window)
    if rope:
        kw["use_rope"] = True
    if num_kv_heads:
        if kw["num_heads"] % int(num_kv_heads):
            raise ValueError(f"{kw['num_heads']} query heads do not split into {num_kv_heads} KV groups")
        kw["num_kv_heads"] = int(num_kv_heads)
    return cls, kw


class AddBranchMixer(FLAMixer):
    """The CKDA mixer plus a parallel fla attention branch: out = mixer(x) + branch(x)."""

    def __init__(self, config, submodules=None, layer_number: int = 1, attn_mask_type=None,
                 arm: str | None = None, role: str = "linear", hybrid_arm: str | None = None,
                 window: int | None = None, rope: bool = False, **kwargs):
        super().__init__(config, submodules=submodules, layer_number=layer_number,
                         attn_mask_type=attn_mask_type, arm=arm, role=role, **kwargs)
        cls, kw = branch_kwargs(hybrid_arm, config, layer_number, window, rope)
        self.branch = cls(**kw)

    def forward(self, hidden_states, *args, **kwargs):
        y, bias = super().forward(hidden_states, *args, **kwargs)
        x = hidden_states.transpose(0, 1).contiguous()
        b = self.branch(x)
        b = b[0] if isinstance(b, tuple) else b
        return y + b.transpose(0, 1).contiguous(), bias


class RopeAttnMixer(FLAMixer):
    """The hybrid's attention layer with use_rope switched on (same weights) and / or rebuilt with fewer KV
    heads (attn_kv; parameter keys unchanged, k_proj / v_proj smaller)."""

    def __init__(self, config, submodules=None, layer_number: int = 1, attn_mask_type=None,
                 arm: str | None = None, role: str = "attn", rope: bool = True,
                 num_kv_heads: int | None = None, **kwargs):
        super(FLAMixer, self).__init__(config=config)
        self.arm = arm
        cls, kw = branch_kwargs(arm, config, layer_number, None, rope, num_kv_heads)
        self.layer = cls(**kw)


def active() -> bool:
    iv = parse()
    return bool(iv["add"] or iv["attn_rope"] or iv["attn_kv"])


def apply(specs: list, config, hybrid_arm: str, every: int | None) -> list:
    """The per-layer specs with the SKDA_INTERVENTION changes applied. ``every``: the hybrid's attention
    period, None for a pure linear stack (then ``hybrid_arm`` is the arm's hybrid sibling, whose attention
    flavor -- fla's gated NoPE MHA -- the branches take)."""
    iv = parse()
    if not iv["add"] and not iv["attn_rope"] and not iv["attn_kv"]:
        return specs
    if iv["attn_kv"] and not every:
        raise ValueError("attn_kv needs a hybrid stack (existing attention layers)")
    branches = branch_layers(iv, config.num_layers, every)
    out = []
    for i, spec in enumerate(specs):
        n = i + 1
        sa = spec.submodules.self_attention
        if n in branches:
            spec = copy.deepcopy(spec)
            spec.submodules.self_attention = ModuleSpec(
                module=AddBranchMixer,
                params={**sa.params, "hybrid_arm": hybrid_arm, "window": branches[n], "rope": iv["add_rope"]})
            spec.submodules.self_attention.submodules = None
        elif every and n % every == 0 and (iv["attn_rope"] or iv["attn_kv"]):
            spec = copy.deepcopy(spec)
            spec.submodules.self_attention = ModuleSpec(
                module=RopeAttnMixer,
                params={**sa.params, "rope": iv["attn_rope"], "num_kv_heads": iv["attn_kv"]})
            spec.submodules.self_attention.submodules = None
        out.append(spec)
    print(f"[recall_surgery] SKDA_INTERVENTION={os.environ.get('SKDA_INTERVENTION', '')!r}: branches "
          f"{ {k: v for k, v in sorted(branches.items())} }, attn_rope={iv['attn_rope']}, "
          f"add_rope={iv['add_rope']}, attn_kv={iv['attn_kv']}", flush=True)
    return out

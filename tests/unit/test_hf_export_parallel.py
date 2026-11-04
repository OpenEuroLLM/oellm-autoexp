"""The parallel exporter's pieces against the sequential exporter's arithmetic."""
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

from oellm_autoexp.hf_export.export_ambitious import fp8_tiles, routed_experts
from oellm_autoexp.hf_export.export_parallel import Unit, _W, _read_unit, plan, write_safetensors
from oellm_autoexp.hf_export.fsdp_dtensor import stack_experts


def _experts(n=5, f=4, h=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = "decoder.layers.1."
    t = {}
    for e in range(n):
        t[f"{p}mlp.experts.linear_fc1.weight{e}_w"] = torch.randn(f, h, generator=g)
        t[f"{p}mlp.experts.linear_fc1.weight{e}_v"] = torch.randn(f, h, generator=g)
        t[f"{p}mlp.experts.linear_fc2.weight{e}"] = torch.randn(h, f * 32, generator=g)
    return p, t


def _stacked(t, p, expert_dtype):
    """The pre-refactor (stacked) conversion."""
    t = dict(t)
    gate = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_w")
    up = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_v")
    experts = {"gate_up_proj": torch.cat([gate, up], dim=1).to(torch.bfloat16)}
    experts["down_proj"] = stack_experts(t, p + "mlp.experts.linear_fc2.weight").to(torch.bfloat16)
    out = {}
    for n, w in experts.items():
        if expert_dtype == "fp8":
            out[n], out[n + "_scale"] = fp8_tiles(w)
        else:
            out[n] = w
    return out


def test_routed_experts_match_stacked():
    for dtype in ("bf16", "fp8"):
        p, t = _experts()
        ref = _stacked(t, p, dtype)
        got = routed_experts(t, p, torch.bfloat16, dtype)
        assert not t
        assert ref.keys() == got.keys()
        for k in ref:
            assert ref[k].dtype == got[k].dtype and ref[k].shape == got[k].shape, k
            assert torch.equal(ref[k].view(torch.uint8), got[k].view(torch.uint8)), k


def test_write_safetensors_round_trip(tmp_path):
    ts = {"a": torch.randn(3, 5), "b": torch.randn(7).to(torch.bfloat16), "c": torch.arange(9, dtype=torch.uint8),
          "d": torch.tensor(2.5), "e": torch.arange(4, dtype=torch.int64)}
    write_safetensors(tmp_path / "x.safetensors", ts)
    with safe_open(tmp_path / "x.safetensors", framework="pt") as f:
        assert set(f.keys()) == set(ts)
        for k, v in ts.items():
            assert torch.equal(f.get_tensor(k), v), k


def test_plan_reads_tables_by_rows(tmp_path):
    """Row-range table units + whole-tensor layer units read the checkpoint's exact values."""
    from torch.distributed.checkpoint.metadata import ChunkStorageMetadata  # noqa: F401
    table = torch.randn(40, 8)
    data = {"model.module.embedding.word_embeddings.weight": torch.randn(6, 4),
            "model.module.decoder.layers.0.table": table,
            "model.module.decoder.layers.0.key_proj.weight": torch.randn(3, 2),
            "optimizer.x": torch.ones(2)}
    dcp.save(data, storage_writer=dcp.FileSystemWriter(tmp_path / "ck"))
    cfg = {"block_types": ["engram"], "engram_table_sizes": [17, 13]}  # 30 of 40 rows exported
    _, units, files = plan(tmp_path / "ck", lambda vocab: cfg, table_unit_bytes=1)
    _W["read_threads"] = 2
    assert len(files) == 3
    tus = [u for u in units if u.kind == "table"]
    assert tus and tus[0].rows[0] == 0 and tus[-1].rows[1] >= 30
    for u in units:
        t = _read_unit(u)
        for name, v in t.items():
            ref = data["model.module." + name]
            if u.kind == "table":
                ref = ref[u.rows[0]:u.rows[1]]
            assert torch.equal(v, ref), name

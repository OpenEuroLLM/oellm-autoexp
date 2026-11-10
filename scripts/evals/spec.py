"""Typed specs of the FLAG eval pipeline (scripts/evals/launch.py, run_job.py).

An ``EvalSpec`` (a YAML file in config/evals/, plus ``--set key=value`` overrides) describes one
checkpoint's evals: the export, the rows, and per harness the engine layout, the model-code
switches and the job limits. launch.py turns it into an oellm-eval plan (submodules/oellm-eval,
oellm/runner: rows -> units) and a runtime for the cluster, one pool job per unit running
``oellm.runner.worker``; the export is the one job of autoexp's own (a ``JobSpec``, run_job.py). Every value is checked against its field's type when it is loaded (unknown
keys, wrong types and values outside a ``Literal`` are errors that name the key), and the
cross-field rules in ``check()`` run before anything is written or submitted.

A YAML value overrides only what it names: a section given in part keeps the rest of that
section's defaults (``lm_eval: {engine: {max_num_seqs: 128}}`` keeps the lm-eval layout TP1 x DP4).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal

Harness = Literal["lm-eval-harness", "lighteval", "evalchemy"]
HARNESSES: tuple[str, ...] = typing.get_args(Harness)
GPUS_PER_NODE = 4

# export dir stem per training run (must match the `name` in export_main_run_v2.sh)
EXPORT_STEMS = {
    "ambitious_v4": "ambitious_v4_2ph",
    "ambitious_v4_hd256": "ambitious_v4_2ph_hd256",
}


class SpecError(ValueError):
    pass


# ---------------------------------------------------------------- strict loading


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if tp is Any:
        return value
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise SpecError(f"{where}: {value!r} is not one of {list(typing.get_args(tp))}")
        return value
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise SpecError(f"{where}: may not be null")
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, where)
            except SpecError as e:
                errors.append(str(e))
        raise SpecError("; ".join(errors))
    if origin is list:
        if not isinstance(value, list):
            raise SpecError(f"{where}: expected a list, got {value!r}")
        (item,) = typing.get_args(tp)
        return [_convert(item, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise SpecError(f"{where}: expected a mapping, got {value!r}")
        k_tp, v_tp = typing.get_args(tp)
        return {_convert(k_tp, k, f"{where} key"): _convert(v_tp, v, f"{where}.{k}") for k, v in value.items()}
    if dataclasses.is_dataclass(tp):
        return load(tp, value, where)
    if tp is bool:
        if not isinstance(value, bool):
            raise SpecError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpecError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpecError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise SpecError(f"{where}: expected a string, got {value!r}")
        return value
    raise SpecError(f"{where}: unsupported field type {_type_name(tp)}")


def load(cls: type, data: Any, where: str = "") -> Any:
    """``cls`` from a mapping: every key a field, every value of the field's type."""
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise SpecError(f"{where or cls.__name__}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - names)
    if unknown:
        raise SpecError(f"{where or cls.__name__}: unknown key(s) {unknown} (known: {sorted(names)})")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        key = f"{where}.{f.name}" if where else f.name
        if f.name in data:
            kwargs[f.name] = _convert(hints[f.name], data[f.name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise SpecError(f"{key}: required")
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    return dataclasses.asdict(obj)


def merge(base: dict, over: dict) -> dict:
    """``over`` onto ``base``, mappings key by key (lists and scalars replace)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_path(data: dict, dotted: str, value: Any) -> dict:
    """``data`` with ``a.b.c = value`` (``--set`` overrides)."""
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        nxt = node.get(k)
        if nxt is None:
            nxt = node[k] = {}
        if not isinstance(nxt, dict):
            raise SpecError(f"--set {dotted}: {k} is not a section")
        node = nxt
    node[keys[-1]] = value
    return data


# ---------------------------------------------------------------- engine and model code


@dataclass
class ModelCode:
    """The oellm_ambitious model-code switches of the vLLM fork (its OELLM_* environment).
    The defaults are the validated fast settings (experiments 008 / 012)."""

    mhc_impl: Literal["triton", "reference"] = "triton"
    fusion_impl: Literal["triton", "reference"] = "triton"
    decode_kernels: Literal["triton", "reference"] = "triton"
    moe_fp8: Literal["w8a16", ""] = "w8a16"
    engram_tables: Literal["uva", "ram", "mmap", "gpu"] = "uva"
    engram_decode_impl: Literal["batched", "reference"] = "batched"
    engram_lookup: Literal["leader", "replicated"] = "leader"
    engram_madvise: Literal["normal", "random", "sequential", "willneed"] = "normal"
    local_expert_load: bool = True
    decode_graphs: bool = False          # the model's own decode graphs (OELLM_DECODE_GRAPHS)
    graph_max_rows: int = 256
    vllm_cudagraph: Literal["full_decode", "none", ""] = "full_decode"
    enforce_eager: bool = False          # vLLM enforce_eager (OELLM_VLLM_ENFORCE_EAGER)

    def env(self) -> dict[str, str]:
        b = lambda x: "1" if x else "0"  # noqa: E731
        return {
            "OELLM_MHC_IMPL": self.mhc_impl,
            "OELLM_FUSION_IMPL": self.fusion_impl,
            "OELLM_DECODE_KERNELS": self.decode_kernels,
            "OELLM_MOE_FP8": self.moe_fp8,
            "OELLM_ENGRAM_TABLES": self.engram_tables,
            "OELLM_ENGRAM_DECODE_IMPL": self.engram_decode_impl,
            "OELLM_ENGRAM_LOOKUP": self.engram_lookup,
            "OELLM_ENGRAM_MADVISE": self.engram_madvise,
            "OELLM_LOCAL_EXPERT_LOAD": b(self.local_expert_load),
            "OELLM_DECODE_GRAPHS": b(self.decode_graphs),
            "OELLM_GRAPH_MAX_ROWS": str(self.graph_max_rows),
            "OELLM_VLLM_CUDAGRAPH": self.vllm_cudagraph,
            "OELLM_VLLM_ENFORCE_EAGER": b(self.enforce_eager),
        }


@dataclass
class Engine:
    """One node's vLLM engine(s): ``tp`` x ``dp`` GPUs."""

    tp: int = 4
    dp: int = 1
    expert_parallel: bool = True
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 32768
    # null: the export's window (max_position_embeddings; vLLM fits it to the KV cache)
    max_model_length: int | None = None
    gpu_memory_utilization: float = 0.9
    # lm-eval only: requests per lm-eval batch, and the durable per-request cache
    request_batch_size: int | None = None
    response_cache: bool = False
    # lm-eval only: engines per node, each on tp x dp GPUs of its own (small dense models)
    replicas: int = 1

    def check(self, where: str) -> None:
        if self.tp < 1 or self.dp < 1 or self.replicas < 1:
            raise SpecError(f"{where}: tp, dp and replicas must be positive")
        if self.tp * self.dp * self.replicas > GPUS_PER_NODE:
            raise SpecError(f"{where}: tp x dp x replicas = {self.tp * self.dp * self.replicas} > "
                            f"{GPUS_PER_NODE} GPUs of a node")
        if self.dp > 1 and not self.expert_parallel:
            raise SpecError(f"{where}: dp > 1 needs expert_parallel (the MoE bound of the model)")
        if self.max_num_seqs < self.dp:
            raise SpecError(f"{where}: max_num_seqs {self.max_num_seqs} < dp {self.dp}")
        if not 0.1 <= self.gpu_memory_utilization <= 0.98:
            raise SpecError(f"{where}: gpu_memory_utilization {self.gpu_memory_utilization} outside [0.1, 0.98]")
        if self.request_batch_size is not None and self.request_batch_size < 1:
            raise SpecError(f"{where}: request_batch_size must be positive")
        if self.response_cache and self.request_batch_size is None:
            raise SpecError(f"{where}: response_cache needs a request_batch_size")


@dataclass
class JobLimits:
    priority: int = 55
    # null: three times the expected time, 2-48 h (grouped harnesses), or 12 h (queue workers)
    time_h: int | None = None
    # also run on the host's excluded (slow but working) nodes
    use_excluded_nodes: bool = True


# ---------------------------------------------------------------- harness sections


def _lm_engine() -> Engine:
    # the fast layout (flag_eval_fast_mn5, experiments 008 / 012): pure expert parallelism over
    # four engines, one copy of the Engram tables in host memory; ~3x the rows/s of TP4
    return Engine(tp=1, dp=4, expert_parallel=True, max_num_seqs=256, max_num_batched_tokens=32768,
                  gpu_memory_utilization=0.85, request_batch_size=256, response_cache=True)


def _grouped_engine(max_model_length: int | None) -> Engine:
    # TP4 (one engine over the node); 64 sequences: full 1012-sample rows OOMed at 256 (~2.5 GiB
    # in CUDA-graph pools)
    return Engine(tp=4, dp=1, expert_parallel=True, max_num_seqs=64, max_num_batched_tokens=32768,
                  max_model_length=max_model_length, gpu_memory_utilization=0.9)


@dataclass
class LmEvalSpec:
    """lm-eval rows through a shared queue (oellm.runner.queue): ``workers`` one-node session units,
    one engine each (``engine.replicas`` engines for small models), items claimed until the queue is
    empty; rows longer than ``split_s`` of engine time become up to ``max_parts`` document parts
    (oellm.runner.lm_eval_parts), merged by lm-eval's own aggregation."""

    enabled: bool = True
    workers: int = 10
    split_s: float = 300.0
    max_parts: int = 8
    engine: Engine = field(default_factory=_lm_engine)
    model_code: ModelCode = field(default_factory=ModelCode)
    job: JobLimits = field(default_factory=lambda: JobLimits(priority=60))

    def check(self) -> None:
        if self.workers < 1:
            raise SpecError("lm_eval.workers must be positive")
        if self.max_parts < 1 or self.split_s < 0:
            raise SpecError("lm_eval: max_parts >= 1 and split_s >= 0")
        if self.engine.max_model_length is not None:
            raise SpecError("lm_eval.engine.max_model_length: the lm-eval session uses the export's window (null)")
        self.engine.check("lm_eval.engine")


@dataclass
class GroupedSpec:
    """lighteval / evalchemy rows in ``groups`` one-node jobs, one model load per job (evalchemy:
    one per task), balanced longest first by the reference times."""

    enabled: bool = True
    groups: int = 1
    # rows per job at most (lighteval writes its results only at the end of a call: a job cut by
    # its wall clock loses all its rows)
    max_rows: int | None = None
    exclude_tasks: list[str] = field(default_factory=list)
    engine: Engine = field(default_factory=lambda: _grouped_engine(None))
    model_code: ModelCode = field(default_factory=lambda: ModelCode(graph_max_rows=64))
    job: JobLimits = field(default_factory=JobLimits)
    # lighteval only: "accelerate" = HF transformers, batch_size 32, as oellm-eval runs lighteval on
    # JUPITER (the backend check; the 84B export does not fit it)
    backend: Literal["vllm", "accelerate"] = "vllm"

    def check(self, name: str) -> None:
        if self.backend != "vllm" and name != "lighteval":
            raise SpecError(f"{name}.backend: only lighteval has an accelerate backend")
        if self.groups < 1:
            raise SpecError(f"{name}.groups must be positive")
        if self.max_rows is not None and self.max_rows < 1:
            raise SpecError(f"{name}.max_rows must be positive")
        if self.engine.request_batch_size is not None or self.engine.response_cache or self.engine.replicas != 1:
            raise SpecError(f"{name}.engine: request_batch_size / response_cache / replicas are lm-eval only")
        self.engine.check(f"{name}.engine")


def _lighteval() -> GroupedSpec:
    return GroupedSpec(groups=7, max_rows=12, engine=_grouped_engine(32768))


def _evalchemy() -> GroupedSpec:
    # LiveCodeBench: datasets.map PicklingError in evalchemy (2026-11-06)
    return GroupedSpec(groups=1, exclude_tasks=["LiveCodeBench"])


# ---------------------------------------------------------------- the launch


@dataclass
class ExportSpec:
    """The HF export of a training checkpoint (export_main_run_v2.sh): made by the launch's
    export job unless <export>/.downstream_ready exists."""

    model: Literal["ambitious_v4", "ambitious_v4_hd256"] = "ambitious_v4_hd256"
    step: int | None = None
    expert_dtype: Literal["fp8"] = "fp8"
    exporter: Literal["parallel", "sequential"] = "parallel"
    # the HF config's max_position_embeddings for these NoPE models (length-extrapolating)
    max_positions: int = 100_000_000
    # an existing export by its directory name instead (no export job when it is ready)
    name: str | None = None
    # a training run export_main_run_v2.sh does not name (e.g. a ckda_branch post-fit arm,
    # production/ckda_branch/<arm>): the export job runs scripts/korbi/hf_export/export_run.sh on it;
    # needs name and step
    run_dir: str | None = None

    @property
    def export_name(self) -> str:
        if self.run_dir and (not self.name or self.step is None):
            raise SpecError("export.run_dir needs export.name and export.step")
        if self.name:
            return self.name
        if self.step is None:
            raise SpecError("export: give step (or name)")
        return f"{EXPORT_STEMS[self.model]}_{self.expert_dtype}_{self.step}"


def _flag_groups() -> dict[str, str]:
    return {"vllm": "flag-evals-vllm", "lighteval": "flag-evals-lighteval"}


@dataclass
class RowSelection:
    # oellm-eval task groups per half (row names <half>/<task>/<n_shot>): the 438 FLAG rows
    groups: dict[str, str] = field(default_factory=_flag_groups)
    # a suite file instead (tasks: [{name, task, n_shot, suite}]; the old flag_evals.yaml)
    suite_file: str | None = None
    # row names (half/task/n_shot); null: every row of the selected harnesses
    names: list[str] | None = None
    harnesses: list[Harness] = field(default_factory=lambda: list(HARNESSES))
    exclude: list[str] = field(default_factory=list)
    limit: int = 0  # documents per row for test runs (0: all)


@dataclass
class Paths:
    hybrid_env: str = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019"
    exports: str = "/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports"
    views: str = "/gpfs/scratch/ehpc533/hybrid_nucleus/views/identity"
    out: str = "/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm"
    logs: str | None = None  # null: <out>/logs
    pool: str = "/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p134c"
    # the oellm-eval stage (evalchemy checkout, lighteval tasks) and the harness libraries
    oellm_stage: str | None = None       # null: <hybrid_env>/oellm_mn5/oellm_mn5_stage
    pylibs_oellm: str | None = None      # null: <hybrid_env>/pylibs_oellm
    pylibs_lighteval: str | None = None  # null: <hybrid_env>/oellm_mn5/pylibs_lighteval
    stanza: str | None = None            # null: <hybrid_env>/stanza_resources

    def resolved(self) -> "Paths":
        r = self.hybrid_env
        return dataclasses.replace(
            self,
            logs=self.logs or f"{self.out}/logs",
            oellm_stage=self.oellm_stage or f"{r}/oellm_mn5/oellm_mn5_stage",
            pylibs_oellm=self.pylibs_oellm or f"{r}/pylibs_oellm",
            pylibs_lighteval=self.pylibs_lighteval or f"{r}/oellm_mn5/pylibs_lighteval",
            stanza=self.stanza or f"{r}/stanza_resources",
        )


@dataclass
class Slurm:
    account: str = "cns144"
    qos: str = "acc_xehpc"
    # the pool job config every unit's sweep point composes (config/ path)
    job_config: str = "experiments/korbi/production/evals/eval_job_mn5"


@dataclass
class RuntimeSpec:
    """How the harnesses run on the cluster's nodes (an oellm-eval runtime.json).

    mn5        the node-local vLLM install (scripts/korbi/vllm/setup_vllm_uv.sh, in job.sh) with the
               oellm-eval stage and the harness libraries of ``paths`` (lighteval / evalchemy)
    apptainer  one image per harness, as JUPITER's oellm-eval images (``images``: lm-eval-harness,
               lighteval, evalchemy), ``apptainer_args`` / ``binds`` / ``env`` as the cluster needs
    """

    kind: Literal["mn5", "apptainer"] = "mn5"
    images: dict[str, str] = field(default_factory=dict)
    apptainer_args: list[str] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    evalchemy_dir: str = "/opt/evalchemy"  # apptainer: the evalchemy checkout inside the image
    hang_minutes: float = 15.0             # end a command silent this long after vLLM's shutdown

    def check(self) -> None:
        if self.kind == "apptainer":
            missing = [h for h in HARNESSES if h not in self.images]
            if missing:
                raise SpecError(f"runtime.images: no image for {missing}")
        if self.hang_minutes < 0:
            raise SpecError("runtime.hang_minutes must not be negative")


@dataclass
class EvalSpec:
    export: ExportSpec = field(default_factory=ExportSpec)
    rows: RowSelection = field(default_factory=RowSelection)
    lm_eval: LmEvalSpec = field(default_factory=LmEvalSpec)
    lighteval: GroupedSpec = field(default_factory=_lighteval)
    evalchemy: GroupedSpec = field(default_factory=_evalchemy)
    paths: Paths = field(default_factory=Paths)
    slurm: Slurm = field(default_factory=Slurm)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    tag: str = ""  # job-name suffix (a second launch of the same export)

    def check(self) -> None:
        _ = self.export.export_name
        if self.tag and not self.tag.replace("_", "").isalnum():
            raise SpecError(f"tag {self.tag!r}: letters, digits and _ only")
        self.lm_eval.check()
        self.lighteval.check("lighteval")
        self.evalchemy.check("evalchemy")
        self.runtime.check()


def default_dict(cls: type = None) -> dict:
    return to_dict((cls or EvalSpec)())


def eval_spec(data: dict | None = None, sets: list[str] | None = None) -> EvalSpec:
    """An ``EvalSpec`` from a YAML mapping and ``key=value`` overrides (values parsed as YAML),
    each onto the defaults, then checked."""
    import yaml

    merged = merge(default_dict(), data or {})
    for s in sets or []:
        key, sep, raw = s.partition("=")
        if not sep:
            raise SpecError(f"--set {s!r}: expected key=value")
        merged = set_path(merged, key.strip(), yaml.safe_load(raw))
    spec = load(EvalSpec, merged)
    spec.check()
    return spec


# ---------------------------------------------------------------- the export job


@dataclass
class JobSpec:
    """The export job of a launch (run_job.py): the HF export of the checkpoint, its identity
    chat-template view, its current model code and <export>/.downstream_ready. The eval units are
    oellm-eval's (plan.json)."""

    kind: Literal["export"]
    name: str
    export_name: str
    export_dir: str
    view_dir: str
    paths: Paths
    export: ExportSpec
    est_s: float = 0.0

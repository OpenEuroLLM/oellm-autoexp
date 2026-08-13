import os
from pathlib import Path

from omegaconf import OmegaConf

from oellm_autoexp.config.resolvers import register_default_resolvers
from oellm_autoexp.slurm_gen.generator import build_sbatch_directives
from oellm_autoexp.slurm_gen.schema import SbatchConfig, SlurmConfig


def test_resolver_join():
    cfg = {"a": ["1", "2", "3"], "c": "${oc.join:'.',${a}}"}

    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["c"] == "1.2.3"


def test_template():
    cfg = {"a": "a", "b": "${oc.tmpl:'$%=\\'$%\\'',${a}}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == "$a='$a'"


def test_template_map():
    cfg = {"a": ["a", "b", "c"], "b": "${oc.maptmpl:'$%=\\'$%\\'',${a}}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == ["$a='$a'", "$b='$b'", "$c='$c'"]


def test_split():
    cfg = {"a": "123 1312 312 13 1241 1214", "b": "${oc.split:${a},' '}"}
    ocfg = OmegaConf.create(cfg)
    cfg = OmegaConf.to_container(ocfg, resolve=True)
    assert cfg["b"] == ["123", "1312", "312", "13", "1241", "1214"]


def test_default_resolvers_evaluate_expressions():
    register_default_resolvers(force=True)

    cfg = OmegaConf.create(
        {
            "product": "${oc.mul:2,3,4}",
            "ceil": "${oc.cdivi:5,2}",
            "difference": "${oc.sub:10,4}",
            "ratio": "${oc.div:9,3}",
            "bool_int": "${oc.int:True}",
        }
    )
    resolved = OmegaConf.to_object(cfg)
    assert resolved["product"] == 24
    assert resolved["ceil"] == 3
    assert resolved["difference"] == 6.0
    assert resolved["ratio"] == 3.0
    assert resolved["bool_int"] == 1


def test_register_default_resolvers_idempotent():
    register_default_resolvers()
    register_default_resolvers()
    cfg = OmegaConf.create({"val": "${oc.addi:1,2}"})
    assert OmegaConf.to_object(cfg)["val"] == 3


def test_exclude_nodes_reads_file(tmp_path: Path):
    register_default_resolvers(force=True)
    listing = tmp_path / "exclude.txt"
    # mix of comments, blank lines, and comma/space separated tokens
    listing.write_text("# bad nodes\nlrdn0417\n\nlrdn0001, lrdn0002\nlrdn0003 lrdn0004\n")
    cfg = OmegaConf.create({"nodes": f"${{oc.exclude_nodes:{listing}}}"})
    assert OmegaConf.to_object(cfg)["nodes"] == "lrdn0417,lrdn0001,lrdn0002,lrdn0003,lrdn0004"


def test_exclude_nodes_dedups_preserving_order(tmp_path: Path):
    register_default_resolvers(force=True)
    listing = tmp_path / "exclude.txt"
    listing.write_text("lrdn0417\nlrdn0001\nlrdn0417\n")
    cfg = OmegaConf.create({"nodes": f"${{oc.exclude_nodes:{listing}}}"})
    assert OmegaConf.to_object(cfg)["nodes"] == "lrdn0417,lrdn0001"


def test_exclude_nodes_missing_or_empty_file_is_none(tmp_path: Path):
    register_default_resolvers(force=True)
    missing = tmp_path / "does_not_exist.txt"
    empty = tmp_path / "empty.txt"
    empty.write_text("# only comments\n\n")
    cfg = OmegaConf.create(
        {
            "missing": f"${{oc.exclude_nodes:{missing}}}",
            "empty": f"${{oc.exclude_nodes:{empty}}}",
        }
    )
    resolved = OmegaConf.to_object(cfg)
    assert resolved["missing"] is None
    assert resolved["empty"] is None


def test_leonardo_autoexclude_config_wires_resolver(tmp_path: Path):
    """The leonardo_autoexclude slurm config derives from leonardo and feeds
    the exclusion file through oc.exclude_nodes into sbatch.exclude."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "config" / "slurm" / "leonardo_autoexclude.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=False)
    assert raw["defaults"][0] == "leonardo"
    exclude_expr = raw["sbatch"]["exclude"]
    assert "oc.exclude_nodes" in exclude_expr

    # Point the resolver at a known file via the configured env var and confirm
    # the value resolves to the recorded node.
    listing = tmp_path / "leonardo_exclude_nodes.txt"
    listing.write_text("lrdn0417\n")
    register_default_resolvers(force=True)
    os.environ["LEONARDO_EXCLUDE_NODES"] = str(listing)
    try:
        resolved = OmegaConf.to_object(OmegaConf.create({"exclude": exclude_expr}))
    finally:
        os.environ.pop("LEONARDO_EXCLUDE_NODES", None)
    assert resolved["exclude"] == "lrdn0417"


def test_exclude_nodes_feeds_sbatch_exclude_directive(tmp_path: Path):
    """End-to-end: a populated list yields an --exclude directive; an empty
    one omits it (resolver returns None -> directive skipped)."""
    register_default_resolvers(force=True)
    listing = tmp_path / "exclude.txt"

    def directives(path: Path) -> list[str]:
        resolved = OmegaConf.to_object(
            OmegaConf.create({"exclude": f"${{oc.exclude_nodes:{path}}}"})
        )
        config = SlurmConfig(
            template_path="templates/base.sbatch",
            script_dir=str(tmp_path),
            log_dir=str(tmp_path),
        )
        config.sbatch = SbatchConfig(exclude=resolved["exclude"])
        return build_sbatch_directives(config)

    # No file yet -> resolver returns None -> no --exclude directive.
    assert not any("--exclude" in d for d in directives(listing))

    # After a node is recorded -> directive appears.
    listing.write_text("lrdn0417\n")
    assert "#SBATCH --exclude=lrdn0417" in directives(listing)

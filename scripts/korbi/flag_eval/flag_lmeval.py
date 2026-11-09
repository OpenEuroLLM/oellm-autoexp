"""Shim: the lm-eval engine session runner is oellm-eval's ``oellm.runner.lm_eval_queue``.

Old command lines keep working: ``--export <model> --out <dir> [--limit l]`` becomes ``--model <model>
--root <dir>/<model name>[_limit<l>]`` (the same results root), ``--suite`` is ignored (row names
carry task and n_shot), FLAG_UNSHARDED_TASKS is still read.

    python scripts/korbi/flag_eval/flag_lmeval.py --export <hf export> --out <dir> --names ... | --queue <dir>
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.korbi.flag_eval import _oellm  # noqa: E402,F401


def translate(argv: list[str]) -> list[str]:
    from oellm.runner.layout import results_root

    out, export, limit, rest = None, None, 0.0, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--export", "--out", "--suite") and i + 1 < len(argv):
            if a == "--export":
                export = argv[i + 1]
            elif a == "--out":
                out = argv[i + 1]
            i += 2
            continue
        if a == "--limit" and i + 1 < len(argv):
            limit = float(argv[i + 1])
        rest.append(a)
        i += 1
    if export is None or out is None:
        return argv  # already the new form (--model / --root)
    root = results_root(out, Path(export).resolve().name, limit)
    return ["--model", export, "--root", str(root), *rest]


def main(argv=None):
    from oellm.runner.lm_eval_queue import main as run

    return run(translate(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()

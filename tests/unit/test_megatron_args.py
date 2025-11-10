import argparse

from oellm_autoexp.backends.megatron_args import (
    build_cmdline_args,
    extract_default_args,
    get_action_specs,
    get_arg_metadata,
)


def test_build_cmdline_args_coercion():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--lr", type=float, default=0.01, dest="lr")
    parser.add_argument("--flags", nargs="+", type=int, default=[1], dest="flags")
    parser.add_argument("--enable", action="store_true", dest="enable")
    parser.add_argument("--mode", action="store_const", const="fast", default="slow", dest="mode")

    metadata = get_arg_metadata(parser)
    action_specs = get_action_specs(parser)
    args = {
        "lr": "0.1",
        "flags": [2, 3],
        "enable": "true",
        "mode": "fast",
    }
    cli = build_cmdline_args(args, metadata, action_specs)
    assert cli == ["--lr", "0.1", "--flags", "2", "3", "--enable", "--mode"]


def test_build_cmdline_args_skips_defaults():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--count", type=int, default=1, dest="count")
    metadata = get_arg_metadata(parser)
    action_specs = get_action_specs(parser)
    cli = build_cmdline_args({"count": 1}, metadata, action_specs)
    assert cli == []


def test_extract_default_args_handles_overrides_and_metadata():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--count", type=int, default=1, dest="count", help="Number of items")
    parser.add_argument(
        "--mode",
        choices=["fast", "slow"],
        default="slow",
        dest="mode",
        help="Execution mode",
    )
    parser.add_argument("--skip", action="store_true", dest="skip")

    metadata = get_arg_metadata(parser)
    defaults = extract_default_args(parser, exclude=["skip"], overrides={"count": 42})

    assert defaults == {"count": 42, "mode": "slow"}
    assert metadata["mode"].choices == ("fast", "slow")
    assert metadata["count"].help == "Number of items"

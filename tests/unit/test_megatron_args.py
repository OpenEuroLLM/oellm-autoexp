import argparse

from oellm_autoexp.backends.megatron_args import build_cmdline_args, get_arg_metadata


def test_build_cmdline_args_coercion():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--lr", type=float, default=0.01, dest="lr")
    parser.add_argument("--flags", nargs="+", type=int, default=[1], dest="flags")
    parser.add_argument("--enable", action="store_true", dest="enable")
    parser.add_argument("--mode", action="store_const", const="fast", default="slow", dest="mode")

    metadata = get_arg_metadata(parser)
    args = {
        "lr": "0.1",
        "flags": [2, 3],
        "enable": "true",
        "mode": "fast",
    }
    cli = build_cmdline_args(args, parser, metadata)
    assert cli == ["--lr", "0.1", "--flags", "2", "3", "--enable", "--mode"]


def test_build_cmdline_args_skips_defaults():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--count", type=int, default=1, dest="count")
    metadata = get_arg_metadata(parser)
    cli = build_cmdline_args({"count": 1}, parser, metadata)
    assert cli == []

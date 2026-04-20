import argparse
import sys
from unittest.mock import MagicMock, patch
import pytest
from argparse_schema import (
    ArgMetadata,
    ActionSpec,
    get_arg_metadata,
    get_action_specs,
    build_cmdline_args,
    extract_default_args,
    generate_dataclass,
    generate_defaults_yaml,
    generate_cli_metadata_code,
)
from argparse_schema.converter import (
    _coerce_value,
    _coerce_arguments,
    _action_type_name,
    _extract_action_type,
    _spec_to_cmdline,
    _ensure_iterable,
)
from typing import Any
from enum import Enum


class MyEnum(Enum):
    VAL1 = "value1"
    VAL2 = "value2"


@pytest.fixture
def mock_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int, default=42, help="Foo help")
    parser.add_argument("--bar", action="store_true", help="Bar help")
    parser.add_argument("--baz", choices=["a", "b"], default="a", help="Baz help")
    parser.add_argument("--list-arg", nargs="+", type=int, default=[1, 2], help="List help")
    parser.add_argument("--choice-enum", type=MyEnum, choices=MyEnum, default=MyEnum.VAL1)
    parser.add_argument("--count", action="count", default=0)
    parser.add_argument("--const", action="store_const", const="fixed", default="default")
    return parser


def test_get_arg_metadata(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    assert "foo" in metadata
    assert metadata["foo"].arg_type is int
    assert metadata["foo"].default == 42
    assert metadata["foo"].help == "Foo help"
    assert metadata["choice_enum"].choices == ("value1", "value2")

    # Test choices normalization for non-Enum
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", choices=[1, 2])
    metadata = get_arg_metadata(parser)
    assert metadata["c"].choices == (1, 2)


def test_get_action_specs(mock_parser):
    specs = get_action_specs(mock_parser)
    assert specs["bar"].action_type == "store_true"
    assert specs["count"].action_type == "count"


def test_extract_default_args(mock_parser):
    # Test with overrides and exclusions
    defaults = extract_default_args(mock_parser, exclude=["foo"], overrides={"bar": True})
    assert "foo" not in defaults
    assert defaults["bar"] is True
    assert defaults["baz"] == "a"


def test_generate_dataclass(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    defaults = extract_default_args(mock_parser)
    code = generate_dataclass(metadata, defaults, class_name="TestConfig")

    assert "class TestConfig(ConfigInterface):" in code
    assert "foo: int = 42" in code
    assert "bar: bool = False" in code
    assert "list_arg: list[int] = field(default_factory=lambda: [1, 2])" in code

    # Keyword exclusion
    metadata_k = {"class": ArgMetadata(arg_type=int, default=1)}
    assert "class:" not in generate_dataclass(metadata_k, {"class": 1})

    # Missing from defaults
    assert "foo:" not in generate_dataclass({"foo": ArgMetadata(default=1)}, {})


def test_generate_dataclass_scalar_list_union():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe", nargs="+", type=str, default=None)
    metadata = get_arg_metadata(parser)

    # Case 1: String default for list arg
    defaults = {"moe": "some_default"}
    code = generate_dataclass(metadata, defaults, allow_scalar_lists=True)
    assert "moe: list[str] | str = 'some_default'" in code

    # Case 2: None default for list arg
    defaults_none = {"moe": None}
    code_none = generate_dataclass(metadata, defaults_none, allow_scalar_lists=True)
    assert "moe: list[str] | None = None" in code_none

    # Case 3: No scalar lists
    code_no_scalar = generate_dataclass(metadata, defaults, allow_scalar_lists=False)
    assert "moe: list[str] = 'some_default'" in code_no_scalar


def test_generate_dataclass_with_extra_fields(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    defaults = extract_default_args(mock_parser)
    extra = {"extra": "dict[str, Any] = field(default_factory=dict)"}

    code = generate_dataclass(metadata, defaults, extra_fields=extra)
    assert "extra: dict[str, Any] = field(default_factory=dict)" in code


def test_generate_defaults_yaml(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    defaults = extract_default_args(mock_parser)
    yaml_text = generate_defaults_yaml(metadata, defaults)

    assert "foo: 42" in yaml_text
    assert "# Foo help" in yaml_text
    assert "baz: a" in yaml_text
    assert "choices: ['a', 'b']" in yaml_text

    # Test include_comments=False
    yaml_no_comments = generate_defaults_yaml(metadata, defaults, include_comments=False)
    assert "# Foo help" not in yaml_no_comments

    # Excluded from YAML
    yaml_ex = generate_defaults_yaml(metadata, defaults, excluded={"foo"})
    assert "foo:" not in yaml_ex

    # Key not in metadata
    yaml_nm = generate_defaults_yaml({}, {"unknown": 1})
    assert "unknown: 1" in yaml_nm


def test_generate_cli_metadata_code(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    specs = get_action_specs(mock_parser)

    # Test with exclusions
    code = generate_cli_metadata_code(metadata, specs, excluded={"foo"})
    assert "'foo'" not in code
    assert "'bar'" in code


def test_build_cmdline_args_list_sep(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    specs = get_action_specs(mock_parser)
    args = {"list_arg": [1, 2, 3]}

    cmdline_space = build_cmdline_args(args, metadata, specs)
    assert cmdline_space == ["--list-arg", "1", "2", "3"]

    cmdline_comma = build_cmdline_args(args, metadata, specs, list_sep=",")
    assert cmdline_comma == ["--list-arg", "1,2,3"]


def test_build_cmdline_args_edge_cases(mock_parser):
    metadata = get_arg_metadata(mock_parser)
    specs = get_action_specs(mock_parser)

    # Count action
    assert build_cmdline_args({"count": 2}, metadata, specs) == ["--count", "--count"]
    assert build_cmdline_args({"count": 0}, metadata, specs) == []
    assert build_cmdline_args({"count": -1}, metadata, specs) == []

    # Count skip defaults (no-op case)
    assert build_cmdline_args({"count": 0}, metadata, specs, skip_defaults=True) == []

    # Const action
    assert build_cmdline_args({"const": "fixed"}, metadata, specs) == ["--const"]
    assert build_cmdline_args({"const": "default"}, metadata, specs) == []

    # List with None or empty
    assert build_cmdline_args({"list_arg": None}, metadata, specs) == []
    assert build_cmdline_args({"list_arg": []}, metadata, specs) == []

    # List matching default
    metadata_ld = {"l": ArgMetadata(arg_type=list, default=["1"])}
    specs_ld = {
        "l": ActionSpec(
            option_strings=("--l",), action_type="store", nargs="+", const=None, default=["1"]
        )
    }
    assert build_cmdline_args({"l": ["1"]}, metadata_ld, specs_ld, skip_defaults=True) == []

    # Missing spec (but in metadata)
    metadata_m = {"a": ArgMetadata(arg_type=int, default=0)}
    assert build_cmdline_args({"a": 1}, metadata_m, {}) == []

    # Store False
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bar", action="store_false", dest="bar", default=True)
    metadata_f = get_arg_metadata(parser)
    specs_f = get_action_specs(parser)
    assert build_cmdline_args({"bar": False}, metadata_f, specs_f) == ["--no-bar"]

    # Skip defaults (standard)
    assert build_cmdline_args({"foo": 42}, metadata, specs, skip_defaults=True) == []

    # Non-default value
    assert build_cmdline_args({"foo": 10}, metadata, specs, skip_defaults=True) == ["--foo", "10"]

    # Empty option strings
    spec_empty = ActionSpec(
        option_strings=(), action_type="store", nargs=None, const=None, default=None
    )
    assert _spec_to_cmdline(spec_empty, 1, False) == []


def test_coerce_value():
    assert _coerce_value("123", int) == 123
    assert _coerce_value("true", bool) is True
    assert _coerce_value("on", bool) is True
    assert _coerce_value("yes", bool) is True
    assert _coerce_value("1", bool) is True
    assert _coerce_value("false", bool) is False
    assert _coerce_value("off", bool) is False
    assert _coerce_value("no", bool) is False
    assert _coerce_value(1, bool) is True
    assert _coerce_value(0, bool) is False
    assert _coerce_value(["1", "2"], list[int]) == [1, 2]
    assert _coerce_value("1,2", list[int]) == [1, 2]
    assert _coerce_value("1", list[int]) == [1]
    assert _coerce_value(1, list) == ["1"]  # elem_type defaults to str
    assert _coerce_value(None, int) is None
    assert _coerce_value(1, None) == 1

    # List with default elem_type str
    assert _coerce_value("a,b", list) == ["a", "b"]

    # Trigger exception in coercion
    assert _coerce_value("not-an-int", int) == "not-an-int"


def test_build_cmdline_args_without_metadata():
    args = {"flag": True, "count": 2}
    assert build_cmdline_args(args, metadata=None, action_specs=None) == [
        "--flag",
        "--count",
        "2",
    ]

    args_list = {"items": None, "off": False}
    assert build_cmdline_args(args_list, metadata=None, action_specs=None, list_sep="=") == []

    args_list_sep = {"items": "a,b"}
    assert build_cmdline_args(args_list_sep, metadata=None, action_specs=None, list_sep="=") == [
        "--items=a,b"
    ]

    args_bool_str = {"flag": "True", "off": "False"}
    assert build_cmdline_args(args_bool_str, metadata=None, action_specs=None) == [
        "--flag",
        "True",
        "--off",
        "False",
    ]


def test_type_repr_edge_cases():
    from argparse_schema.codegen import _type_repr, _literal_token, _type_name
    from argparse_schema import ArgMetadata

    # Choices mismatch
    meta = ArgMetadata(choices=(1, 2), arg_type=int, default=1)
    assert _type_repr(meta, default="1")[0] == "str"

    # Choices with None
    meta_cn = ArgMetadata(choices=(1, None), arg_type=int, default=None)
    assert _type_repr(meta_cn, default=None)[0] == "Literal[1, None]"

    # Any type
    meta_any = ArgMetadata(default=None)
    assert "Any" in _type_repr(meta_any, default=None)[0]

    # Meta default fallback
    meta_def = ArgMetadata(default=10, arg_type=int)
    assert _type_repr(meta_def, default=None)[0] == "int"

    # Float
    meta_float = ArgMetadata(arg_type=float, default=0.0)
    assert _type_repr(meta_float, default=0.0)[0] == "float"

    # Element type Any for list (default fallback)
    meta_list = ArgMetadata(arg_type=list, default=None)
    assert "list[Any]" in _type_repr(meta_list, default=None)[0]

    # List with custom elem_type
    meta_list_c = ArgMetadata(arg_type=list, element_type=dict, default=None)
    assert "list[Any]" in _type_repr(meta_list_c, default=None)[0]

    # Literal token
    assert _literal_token(None) == "None"
    assert _literal_token("s") == "'s'"
    assert _literal_token(1) == "1"

    # Type name
    assert _type_name(None) == "None"
    assert _type_name(Any) == "Any"
    assert _type_name(int) == "int"
    assert _type_name(1, default="Custom") == "Custom"


def test_codegen_format_default():
    from argparse_schema.codegen import _format_default

    assert _format_default("s")[0] == "'s'"
    assert _format_default(True)[0] == "True"
    assert _format_default(False)[0] == "False"
    assert _format_default(None)[0] == "None"
    assert "default_factory" in _format_default({"a": 1})[0]


def test_generate_defaults_yaml_skips_non_kv_lines():
    metadata = {"foo": ArgMetadata(default=None, help="Foo help")}
    defaults = {"foo": 1}

    with patch("argparse_schema.codegen.OmegaConf.to_yaml") as mock_to_yaml:
        mock_to_yaml.return_value = "\n# comment\nfoo: 1\n"
        yaml_text = generate_defaults_yaml(metadata, defaults)

    assert "# comment" in yaml_text
    assert "foo: 1  # Foo help" in yaml_text


def test_ensure_iterable():
    assert _ensure_iterable(1) == [1]
    assert _ensure_iterable([1]) == [1]


def test_action_type_name():
    mock_action = MagicMock()
    mock_action.__class__.__name__ = "_AppendAction"
    assert _action_type_name(mock_action) == "append"

    mock_action.__class__.__name__ = "_AppendConstAction"
    assert _action_type_name(mock_action) == "append_const"

    mock_action.__class__.__name__ = "_StoreFalseAction"
    assert _action_type_name(mock_action) == "store_false"

    mock_action.__class__.__name__ = "_StoreConstAction"
    assert _action_type_name(mock_action) == "store_const"

    mock_action.__class__.__name__ = "UnknownAction"
    assert _action_type_name(mock_action) == "store"


def test_coerce_arguments_missing_key():
    metadata = {"a": ArgMetadata(arg_type=int, default=0)}
    assert _coerce_arguments({"b": 1}, metadata) == {}


def test_extract_default_args_enum():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enum", type=MyEnum, choices=MyEnum, default=MyEnum.VAL1)
    defaults = extract_default_args(parser)
    assert defaults["enum"] == "VAL1"


def test_extract_action_type():
    mock_action = MagicMock(spec=argparse.Action)
    mock_action.nargs = "+"
    assert _extract_action_type(mock_action) is list

    mock_action.nargs = None
    mock_action.type = int
    assert _extract_action_type(mock_action) is int


def test_register_argparse_resolver():
    from argparse_schema.resolver import register_argparse_resolver
    from omegaconf import OmegaConf, DictConfig

    metadata = {"foo": ArgMetadata(arg_type=int, default=1)}
    specs = {
        "foo": ActionSpec(
            option_strings=("--foo",),
            action_type="store",
            nargs=None,
            const=None,
            default=1,
        )
    }

    with patch.object(OmegaConf, "register_new_resolver") as register:
        register_argparse_resolver(
            name="test",
            arg_metadata=metadata,
            action_specs=specs,
            skip_defaults=False,
        )
        name_arg, resolver_func = register.call_args[0]
        assert name_arg == "oc.test"
        result = resolver_func(DictConfig({"foo": 2}))
        assert list(result) == ["--foo", "2"]


def test_get_megatron_parser_success():
    mock_megatron = MagicMock()
    mock_add_args = MagicMock(return_value=argparse.ArgumentParser())
    mock_megatron.training.arguments.add_megatron_arguments = mock_add_args

    with patch.dict(
        sys.modules,
        {
            "megatron": mock_megatron,
            "megatron.training": mock_megatron.training,
            "megatron.training.arguments": mock_megatron.training.arguments,
        },
    ):
        from megatron.training.arguments import add_megatron_arguments

        parser = argparse.ArgumentParser()
        add_megatron_arguments(parser)
        mock_add_args.assert_called_once()

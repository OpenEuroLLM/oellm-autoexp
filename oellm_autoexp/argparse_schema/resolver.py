from omegaconf import OmegaConf, DictConfig, ListConfig
from .converter import ActionSpec, ArgMetadata, build_cmdline_args


def register_argparse_resolver(
    name: str = "argres",
    *,
    arg_metadata: dict[str, ArgMetadata] | None = None,
    action_specs: dict[str, ActionSpec] | None = None,
    skip_defaults: bool = True,
    list_sep: str | None = None,
):
    def argparse_resolver(args_dict: DictConfig) -> ListConfig:
        OmegaConf.resolve(args_dict)
        return ListConfig(
            build_cmdline_args(
                args=OmegaConf.to_container(args_dict),
                metadata=arg_metadata,
                action_specs=action_specs,
                skip_defaults=skip_defaults,
                list_sep=list_sep,
            )
        )

    OmegaConf.register_new_resolver("oc." + name, argparse_resolver)

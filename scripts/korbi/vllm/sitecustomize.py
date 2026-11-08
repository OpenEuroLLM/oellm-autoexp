# Installed into the node-local vLLM tree ($DEST, first on PYTHONPATH) by setup_vllm_uv.sh and
# build_pylibs_vllm.sh.
# $DEST's torch 2.13 replaces the image's, so the image's optional packages built against the
# image's torch / NVSHMEM / Triton fail to import: moonep/_C (undefined c10::cuda symbol),
# deep_ep_cpp (undefined nvshmem_selected_device_transport) and triton_kernels (cannot import
# is_hip_gfx1250). vLLM probes them on every worker start and logs each failure as an error,
# while none is used here (an import that fails cannot be). A None entry in sys.modules makes
# importlib.util.find_spec return None and `import` raise ImportError, so vLLM sees them as
# not installed. OELLM_VLLM_HIDE_MODULES overrides the list (empty: hide nothing).
#
# FlashInfer's JIT cache moves from $HOME (GPFS, shared by all jobs) into this tree: every job
# installs the tree afresh, its sources are newer than any earlier build, so FlashInfer rebuilt
# sampling.so in the shared cache whenever a job started -- and a job starting at the same time
# loaded the half-written library ("sampling.so: file too short", flag_fast 9814 s3, 2026-11-05).
# Per tree: one ~1 min build per node and job, under FlashInfer's lock on node-local /tmp.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", _HERE)

_HIDE = os.environ.get(
    "OELLM_VLLM_HIDE_MODULES", "moonep,deep_ep,triton_kernels,vllm.third_party.triton_kernels"
)
for _name in filter(None, (m.strip() for m in _HIDE.split(","))):
    sys.modules.setdefault(_name, None)


def _nltk_local_index() -> None:
    """nltk.download() fetches its package index from GitHub before it looks at the installed
    data: without internet (MareNostrum 5 compute nodes) every call logged "Error loading
    punkt_tab: <urlopen error>", ~10^5 lines per eval job. With an index.xml in an NLTK_DATA
    directory the downloader reads that file instead, and an installed package is reported up
    to date offline."""
    index = next(
        (
            os.path.join(d, "index.xml")
            for d in os.environ.get("NLTK_DATA", "").split(os.pathsep)
            if d and os.path.isfile(os.path.join(d, "index.xml"))
        ),
        None,
    )
    if index is None:
        return
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "nltk.downloader":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                default, url = module.Downloader.DEFAULT_URL, "file://" + index
                module.Downloader.DEFAULT_URL = url
                # nltk.download is the method of a Downloader made while the module ran
                shared = getattr(module, "_downloader", None)
                if shared is not None and getattr(shared, "_url", None) == default:
                    shared._url = url

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


_nltk_local_index()


def _chain() -> None:
    """Run the sitecustomize this file shadows (the image's /usr/lib/python3.12/sitecustomize.py)."""
    from importlib.machinery import PathFinder

    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]
    spec = PathFinder.find_spec("sitecustomize", rest)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return
    with open(spec.origin) as fh:
        code = compile(fh.read(), spec.origin, "exec")
    exec(code, {"__name__": "sitecustomize", "__file__": spec.origin})


_chain()

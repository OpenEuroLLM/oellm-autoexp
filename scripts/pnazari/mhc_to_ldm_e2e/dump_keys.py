"""Every entry of a torch DCP checkpoint: key, kind, shape, dtype (sorted).   dump_keys.py <iter dir>"""
import sys

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

md = FileSystemReader(sys.argv[1]).read_metadata()
for key in sorted(md.state_dict_metadata):
    meta = md.state_dict_metadata[key]
    if isinstance(meta, TensorStorageMetadata):
        print(key, "tensor", tuple(meta.size), str(meta.properties.dtype).replace("torch.", ""))
    else:
        print(key, type(meta).__name__)

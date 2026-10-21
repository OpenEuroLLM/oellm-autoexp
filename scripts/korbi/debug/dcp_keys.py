"""List the keys (and shapes) of a torch_dist checkpoint matching a substring: dcp_keys.py <ckpt dir> <substr>"""
import sys
from torch.distributed.checkpoint import FileSystemReader
md = FileSystemReader(sys.argv[1]).read_metadata()
for k, v in md.state_dict_metadata.items():
    if sys.argv[2] in k:
        print(k, tuple(getattr(v, "size", ())))

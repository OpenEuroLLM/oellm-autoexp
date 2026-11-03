"""Fail before a model load if Nsight has not captured real CUDA activity."""
import json
from pathlib import Path
import sqlite3
import sys

path = Path(sys.argv[1])
con = sqlite3.connect(path)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables, sorted(tables)
rows = con.execute('SELECT deviceId,COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId').fetchall()
assert len(rows) == 4 and all(row[1] >= 16 and row[2] > 0 for row in rows), rows
result = {'passed': True, 'gpu_kernel_counts_and_ns': rows, 'sqlite': str(path)}
path.with_suffix('.check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

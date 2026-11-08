"""Time a child command without depending on GNU time in the training container."""
import datetime
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
started = time.perf_counter()
rc = subprocess.call(sys.argv[2:])
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
result = {
    'command': sys.argv[2:], 'started_at': started_at,
    'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'wall_seconds': time.perf_counter() - started,
    'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime,
    'max_rss_kib': usage.ru_maxrss, 'exit_code': rc,
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
sys.exit(rc)

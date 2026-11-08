"""One scheduling pass for an original plan containing one unsubmitted new job."""
import sys
from oellm_autoexp.monitor.loop import JobFileStore,MonitorLoop
from oellm_autoexp.monitor.slurm_client import SlurmClient,SlurmClientConfig

store=JobFileStore(sys.argv[1])
records=store.load_all()
assert len(records)==1 and not records[0].runtime.submitted
assert records[0].runtime.runtime_job_id is None
assert records[0].definition.pool_dir is None
assert records[0].definition.start_condition is None
assert records[0].definition.cancel_condition is None
assert records[0].definition.finish_condition is None
loop=MonitorLoop(store=store,slurm_client=SlurmClient(SlurmClientConfig()))
loop.observe_once()
for record in store.load_all():
    print(record.runtime.model_dump() if hasattr(record.runtime,'model_dump') else record.runtime)

# Pool adoption hook (oellm_autoexp/monitor/pool.py, `pool_ctl adopt`), installed as
# <job WorkDir>/sitecustomize.py. Python imports it at startup in every process whose sys.path
# contains that directory -- the adopted job's training step runs `torchrun` there with `.` on
# PYTHONPATH, so it runs in the torchrun agents before any rank, Megatron import or CUDA call.
#
# FAIL-CLOSED: the adopted SLURM job ids are written into THIS file at install. In any process of
# such a job that is not a pool run (AUTOEXP_RUN_ID unset), the hook never returns: it holds the
# process idle (no GPU use) until the pool supervisor attaches and later releases the allocation,
# and exits 0 if no supervisor attaches within HOLD_MAX_S. Either way the job's own training never
# starts. Every other process (other jobs, login-node tools, pool runs) is unaffected: the original
# sitecustomize further down sys.path (the container image's) is chained in.
import os as _os
import sys as _sys

_ADOPTED = {__ADOPTED_JOB_IDS__}  # noqa: F821 (rendered by pool_ctl adopt)
_ADOPT_DIR = __ADOPT_DIR__  # noqa: F821
_HOLD_MAX_S = __HOLD_MAX_S__  # noqa: F821


def _chain_original():
    import importlib.util

    here = _os.path.dirname(_os.path.abspath(__file__))
    for entry in _sys.path:
        try:
            path = _os.path.join(_os.path.abspath(entry or "."), "sitecustomize.py")
        except Exception:
            continue
        if _os.path.dirname(path) == here or not _os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location("_original_sitecustomize", path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return


def _hold(job_id):
    import signal
    import socket
    import time

    def say(msg):
        _sys.stderr.write(f"[pool-adopt] {time.strftime('%Y-%m-%dT%H:%M:%S')} {socket.gethostname()} "
                          f"pid {_os.getpid()}: {msg}\n")
        _sys.stderr.flush()

    def leave(code, msg):
        say(msg)
        _os._exit(code)

    signal.signal(signal.SIGTERM, lambda *_: leave(0, "SIGTERM: leaving"))
    signal.signal(signal.SIGUSR2, lambda *_: leave(0, "SIGUSR2: leaving"))
    t0 = time.time()
    try:
        _os.makedirs(_ADOPT_DIR, exist_ok=True)
        # the first process of the job records when (and where) the allocation started
        fd = _os.open(_os.path.join(_ADOPT_DIR, f"{job_id}.started"), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.write(fd, f"{t0:.0f} {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0))} "
                      f"{socket.gethostname()} {_os.environ.get('SLURM_JOB_NODELIST', '')}\n".encode())
        _os.close(fd)
    except FileExistsError:
        pass
    except Exception as exc:  # never fall through to the job's own code
        say(f"could not write the start marker ({exc}); holding anyway")
    say(f"job {job_id} is adopted by a pool: its own step is held (argv {_sys.argv[:2]})")
    attached = _os.path.join(_ADOPT_DIR, f"{job_id}.attached")
    release = _os.path.join(_ADOPT_DIR, f"{job_id}.release")
    while True:
        try:
            if _os.path.exists(release):
                leave(0, "the pool released the allocation")
            if not _os.path.exists(attached) and time.time() - t0 > _HOLD_MAX_S:
                try:
                    open(_os.path.join(_ADOPT_DIR, f"{job_id}.abandoned"), "a").close()
                except Exception:
                    pass
                leave(0, f"no pool supervisor attached within {_HOLD_MAX_S}s: leaving (no training)")
        except SystemExit:
            raise
        except Exception as exc:
            say(f"poll error ({exc}); holding")
        time.sleep(10)


_job = _os.environ.get("SLURM_JOB_ID", "")
if _job in _ADOPTED and not _os.environ.get("AUTOEXP_RUN_ID"):
    _hold(_job)
_chain_original()

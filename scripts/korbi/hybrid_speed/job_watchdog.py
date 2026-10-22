#!/usr/bin/env python3
"""In-job watchdog for speed / smoke runs: GPU memory trace + hang kill.

Runs on the host (python3 >= 3.6, no dependencies), one instance per node, launched in the
background by templates/marenostrum_watchdog.sbatch next to the training step.

Every node: samples nvidia-smi (memory used / total, utilisation) every WATCHDOG_INTERVAL_S
into <dir>/mem_<host>.csv and keeps the per-GPU peak; host memory (used = total - available)
into <dir>/host_<host>.csv.

Node 0 also follows the job's stdout log (Megatron's "iteration N/" lines). The job is stuck
when no iteration appears within WATCHDOG_STARTUP_S of the start, or the iteration counter
does not move for WATCHDOG_STALL_S. Then node 0 writes <dir>/STALL; every node dumps the
py-spy stacks of its Python processes (all threads, native frames) into
<dir>/stacks_<host>.txt; node 0 prints one WATCHDOG line per node to the job log (last
iteration, peak / current memory, whether any GPU was near-full: the "too much memory"
signature is memory >= WATCHDOG_MEM_FULL_FRAC with utilisation 0) and scancels the job.

Bad-GPU nodes (WATCHDOG_EXCLUDE_FILE set, config/slurm/marenostrum_autoexclude.yaml): every
node records ITSELF in the shared exclusion file, which the next submissions read
(oc.exclude_nodes), when
* its GPU probe at the start sees fewer than WATCHDOG_GPUS GPUs (nvidia-smi fails or lists
  fewer) -- the node behind torchrun's "AssertionError: Invalid device id", or
* the job log shows a hardware-signature CUDA error ("unspecified launch failure", "CUDA error:
  unknown error", an uncorrectable ECC error, "fallen off the bus") on its ranks
  (rank // WATCHDOG_GPUS == SLURM_NODEID: torchrun --node-rank $SLURM_NODEID) and on no other
  node's, and no MoonEP barrier trap was logged: a trap makes every waiting rank report a
  launch failure, so neither a trap nor errors on several nodes implicate one node.
A healthy node failing a code bug does not match either. Each entry is a "# date reason job"
comment line plus the host name.

Env: WATCHDOG_DIR (default <log dir>/watchdog), WATCHDOG_STARTUP_S (1200), WATCHDOG_STALL_S
(300), WATCHDOG_INTERVAL_S (15), WATCHDOG_MEM_FULL_FRAC (0.97), WATCHDOG_PYSPY (path to the
py-spy binary; optional), WATCHDOG_DRY_RUN=1 (report, never cancel), WATCHDOG_EXCLUDE_FILE
(bad-node list; unset: no recording), WATCHDOG_GPUS (4).
"""
import os
import re
import signal
import subprocess
import sys
import time

ITER_RE = re.compile(rb"iteration\s+(\d+)/\s*\d+")
HW_RE = re.compile(rb"\[rank(\d+)\]:[^\n]*(unspecified launch failure|CUDA error: unknown error|"
                   rb"uncorrectable ECC error|fallen off the bus)")
TRAP_RE = re.compile(rb"MoonEP [a-z_ ]*barrier timeout[^\n]*trapping")


def record_bad_node(path, host, reason):
    """Append ``host`` (with a comment line) to the exclusion file unless it is listed."""
    try:
        with open(path) as f:
            listed = {l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")}
    except OSError:
        listed = set()
    if host in listed:
        return False
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "?")
    entry = "# %s %s (job %s)\n%s\n" % (time.strftime("%Y-%m-%d %H:%M"), reason, job, host)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)  # one atomic append
    try:
        os.write(fd, entry.encode())
    finally:
        os.close(fd)
    return True


def gpu_probe(expected):
    """None if ``expected`` GPUs answer nvidia-smi, else the failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, timeout=60).decode()
    except Exception as e:  # nvidia-smi failing is itself the finding
        return "nvidia-smi failed: %s" % str(e).splitlines()[0][:120]
    n = len([l for l in out.splitlines() if l.strip()])
    return None if n >= expected else "GPU probe: %d of %d GPUs" % (n, expected)


class HardwareScan:
    """Follows the job log for hardware-signature CUDA errors (see the module docstring)."""

    def __init__(self, log, gpus):
        self.log, self.gpus, self.offset = log, gpus, 0
        self.nodes, self.first, self.trap = {}, {}, False

    def update(self):
        try:
            with open(self.log, "rb") as f:
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return
        cut = data.rfind(b"\n") + 1  # whole lines only
        self.offset += cut
        data = data[:cut]
        if TRAP_RE.search(data):
            self.trap = True
        for m in HW_RE.finditer(data):
            node = int(m.group(1)) // self.gpus
            self.nodes[node] = self.nodes.get(node, 0) + 1
            self.first.setdefault(node, m.group(2).decode())

    def implicates(self, node_id):
        return not self.trap and set(self.nodes) == {node_id}


def env_f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def job_stdout():
    job = os.environ.get("SLURM_JOB_ID", "")
    try:
        out = subprocess.check_output(["scontrol", "show", "job", job, "-o"]).decode()
    except Exception:
        return None
    m = re.search(r"StdOut=(\S+)", out)
    if not m:
        return None
    path = m.group(1)
    # array jobs: scontrol prints the pattern for pending tasks, the expanded path once running
    path = path.replace("%A", os.environ.get("SLURM_ARRAY_JOB_ID", job))
    path = path.replace("%a", os.environ.get("SLURM_ARRAY_TASK_ID", ""))
    path = path.replace("%j", job)
    return path


def gpu_sample():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=30).decode()
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                rows.append(tuple(int(float(p)) for p in parts))
            except ValueError:
                pass
    return rows


def host_sample():
    """(used, total) host memory in MiB from /proc/meminfo (used = total - available)."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        total = int(info["MemTotal"].split()[0]) // 1024
        return total - int(info["MemAvailable"].split()[0]) // 1024, total
    except (OSError, KeyError, ValueError):
        return 0, 0


def last_iteration(path, state):
    """Scan the log incrementally; return the largest iteration seen so far."""
    try:
        with open(path, "rb") as f:
            f.seek(state.get("pos", 0))
            data = f.read()
            state["pos"] = f.tell()
    except OSError:
        return state.get("iter")
    for m in ITER_RE.finditer(state.get("tail", b"") + data):
        it = int(m.group(1))
        if state.get("iter") is None or it > state["iter"]:
            state["iter"] = it
    state["tail"] = data[-200:]
    return state.get("iter")


def dump_stacks(out_path, pyspy):
    if not pyspy or not os.path.exists(pyspy):
        return
    try:
        pids = subprocess.check_output(["pgrep", "-u", str(os.getuid()), "-f", "python"]).decode().split()
    except Exception:
        pids = []
    me = str(os.getpid())
    with open(out_path, "w") as f:
        for pid in pids:
            if pid == me:
                continue
            try:
                cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b" ")[:300]
            except OSError:
                continue
            if b"job_watchdog" in cmd or b"pretrain" not in cmd:
                continue
            f.write("=== pid %s: %s\n" % (pid, cmd.decode(errors="replace")))
            f.flush()
            try:
                f.write(subprocess.check_output([pyspy, "dump", "--native", "--pid", pid],
                                                stderr=subprocess.STDOUT, timeout=60).decode(errors="replace"))
            except Exception as e:  # noqa: BLE001
                f.write("py-spy failed: %r\n" % (e,))
            f.write("\n")


def main():
    host = os.uname()[1].split(".")[0]
    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    log = job_stdout()
    wdir = os.environ.get("WATCHDOG_DIR") or os.path.join(os.path.dirname(log or "."), "watchdog")
    wdir = os.path.join(wdir, os.environ.get("SLURM_JOB_ID", "nojob"))
    os.makedirs(wdir, exist_ok=True)
    interval = env_f("WATCHDOG_INTERVAL_S", 15)
    startup = env_f("WATCHDOG_STARTUP_S", 1200)
    stall = env_f("WATCHDOG_STALL_S", 300)
    full_frac = env_f("WATCHDOG_MEM_FULL_FRAC", 0.97)
    pyspy = os.environ.get("WATCHDOG_PYSPY")
    dry = os.environ.get("WATCHDOG_DRY_RUN", "0") == "1"
    stall_flag = os.path.join(wdir, "STALL")
    csv = open(os.path.join(wdir, "mem_%s.csv" % host), "a")
    host_csv = open(os.path.join(wdir, "host_%s.csv" % host), "a")  # time, used MiB, total MiB
    peak = {}
    t0 = time.time()
    state = {}
    last_it, last_change = None, t0
    dumped = False
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    exclude_file = os.environ.get("WATCHDOG_EXCLUDE_FILE")
    gpus = int(env_f("WATCHDOG_GPUS", 4))
    scan, seen_at, recorded = (HardwareScan(log, gpus) if exclude_file and log else None), None, False
    if exclude_file:
        problem = gpu_probe(gpus)
        if problem:
            record_bad_node(exclude_file, host, problem)
            if log:
                with open(log, "a") as f:
                    f.write("\nWATCHDOG: %s on %s -> recorded in %s\n" % (problem, host, exclude_file))
            recorded = True
    while not stop:
        now = time.time()
        rows = gpu_sample()
        for idx, used, total, util in rows:
            csv.write("%d,%d,%d,%d,%d\n" % (int(now), idx, used, total, util))
            peak[idx] = max(peak.get(idx, 0), used)
        csv.flush()
        host_csv.write("%d,%d,%d\n" % ((int(now),) + host_sample()))
        host_csv.flush()
        if scan is not None and not recorded:
            scan.update()
            if node_id in scan.nodes and seen_at is None:
                seen_at = now
            # one more look after a grace period: a MoonEP trap or other nodes' errors may follow
            if seen_at is not None and now - seen_at >= min(interval, 10):
                if scan.implicates(node_id):
                    reason = "%s on rank(s) of node %d" % (scan.first[node_id], node_id)
                    record_bad_node(exclude_file, host, reason)
                    with open(log, "a") as f:
                        f.write("\nWATCHDOG: %s (%s) -> recorded in %s\n" % (host, reason, exclude_file))
                recorded = True
        if os.path.exists(stall_flag) and not dumped:
            dump_stacks(os.path.join(wdir, "stacks_%s.txt" % host), pyspy)
            dumped = True
        if node_id == 0 and log:
            it = last_iteration(log, state)
            if it != last_it:
                last_it, last_change = it, now
            waited = now - last_change
            stuck = (last_it is None and waited > startup) or (last_it is not None and waited > stall)
            if stuck and not os.path.exists(stall_flag):
                full = [r for r in rows if r[2] and r[1] >= full_frac * r[2]]
                idle = all(r[3] == 0 for r in rows) if rows else False
                with open(stall_flag, "w") as f:
                    f.write("%d\n" % int(now))
                dump_stacks(os.path.join(wdir, "stacks_%s.txt" % host), pyspy)
                dumped = True
                time.sleep(max(60.0, 3 * interval))  # other nodes dump their stacks
                verdict = "memory-saturated" if full else "not-memory"
                with open(log, "a") as f:
                    f.write("\nWATCHDOG: no progress for %.0f s (last iteration %s, %s, GPUs idle=%s) -> "
                            "cancelling job %s; stacks/memory in %s\n"
                            % (waited, last_it, verdict, idle, os.environ.get("SLURM_JOB_ID"), wdir))
                    for name in sorted(os.listdir(wdir)):
                        if name.startswith("mem_") and name.endswith(".csv"):
                            f.write("WATCHDOG_MEM %s %s\n" % (name[4:-4], summarize(os.path.join(wdir, name))))
                if not dry:
                    job = os.environ.get("SLURM_ARRAY_JOB_ID")
                    job = "%s_%s" % (job, os.environ["SLURM_ARRAY_TASK_ID"]) if job else os.environ.get("SLURM_JOB_ID")
                    subprocess.call(["scancel", job])
        time.sleep(min(interval, 5) if seen_at is not None and not recorded else interval)
    if scan is not None and not recorded:  # the job is ending: decide on what the log holds now
        scan.update()
        if scan.implicates(node_id):
            reason = "%s on rank(s) of node %d" % (scan.first[node_id], node_id)
            record_bad_node(exclude_file, host, reason)
    with open(os.path.join(wdir, "peak_%s.txt" % host), "w") as f:
        f.write(" ".join("gpu%d=%dMiB" % (k, v) for k, v in sorted(peak.items())) + "\n")


def summarize(path):
    """Peak and last memory per GPU from a node's csv."""
    peak, last, total = {}, {}, {}
    try:
        for line in open(path):
            ts, idx, used, tot, util = (int(x) for x in line.split(","))
            peak[idx] = max(peak.get(idx, 0), used)
            last[idx] = (used, util)
            total[idx] = tot
    except (OSError, ValueError):
        return "unreadable"
    return " ".join("gpu%d peak %.1f/%.1f GiB last %.1f GiB util %d%%"
                    % (i, peak[i] / 1024, total[i] / 1024, last[i][0] / 1024, last[i][1]) for i in sorted(peak))


if __name__ == "__main__":
    sys.exit(main())

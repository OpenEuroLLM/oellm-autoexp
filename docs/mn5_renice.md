# MN5 re-nice test: does changing Nice move priority, and does it reset anything? (2026-11-18)

Test job: `48941057` (`mn5_pool_o134c_host003`), partition `acc`, QOS `acc_xehpc`,
account `cns144`, 134 nodes / 536 GPUs, `TimeLimit=1-00:00:00`.
Cluster: `marenostrum_new` (alogin2.bsc.es), Slurm 25.05.9.

Question: can we park queued jobs with a high nice value and promote them later, without
losing queue position?

Answer: nice is a pure, reversible priority offset. It resets nothing. But it cannot
prevent a job from starting, and we cannot push priority above the natural value.

## Baseline (2026-11-18T16:34, before any change)

```
Priority=91228 Nice=0 Account=cns144 QOS=acc_xehpc
JobState=PENDING Reason=QOSMaxJobsPerUserLimit
SubmitTime=2026-11-11T23:45:41 EligibleTime=2026-11-11T23:45:41
AccrueTime=2026-11-11T23:45:41
Requeue=0 Restarts=0
```

`sprio -l` decomposes priority additively:

```
PRIORITY = AGE + ASSOC + FAIRSHARE + JOBSIZE + PARTITION + QOS + NICE
   91194 = 66972 +   0   +   23568  +   154   +    0      + 500 + 0
```

Relevant `scontrol show config` values:

| Parameter | Value |
|---|---|
| `PriorityType` | `priority/multifactor` |
| `PriorityDecayHalfLife` | `00:00:00` (no decay) |
| `PriorityMaxAge` | `10-00:00:00` |
| `PriorityCalcPeriod` | `00:05:00` |
| `PriorityWeightAge` | 100000 |
| `PriorityWeightFairShare` | 100000 |
| `PriorityWeightJobSize` | 10000 |
| `PriorityWeightQOS` | 1000000 |
| `PriorityWeightAssoc`, `PriorityWeightPartition` | 0 |
| `PreemptMode`, `PreemptType` | `OFF`, `(null)` |

There is no `PriorityWeightNice` parameter. Nice enters the sum directly, one to one.
QOS `acc_xehpc`: `Priority=50 MaxJobsPU=3 MaxSubmitPU=366 Flags=DenyOnLimit`.

## Measurements

Only the `Nice` field of job `48941057` was changed, with `scontrol update JobId=48941057 Nice=<n>`.

| Nice | Priority after update | sprio NICE | Result |
|---|---|---|---|
| 0 (start) | 91228 | 0 | baseline |
| 1000 | 90232 | 1000 | exactly -1000; visible ~2 s after the update |
| -20 | update rejected | - | `Access/permission denied for job 48941057` |
| 300000 | 1 | 300000 | clamped to the floor of 1 |
| 2147483645 | 1 | 2147483645 | same floor, no further effect |
| 0 (restored) | 91245 | 0 | baseline + 17 = age accrual only |

Notes:

- The new priority appears immediately. We do not wait for `PriorityCalcPeriod`.
- The offset is linear and 1:1. Maximum possible priority is
  `100000 + 100000 + 10000 + 500 = 210500`, so any `Nice >= 250000` reaches the floor.
- Slurm priority bottoms out at 1, not 0. Values above 2147483645 are rejected.
- Negative nice needs root. `man sbatch` states: "Only privileged users can specify a
  negative adjustment" (range +-2147483645). So the only user lever is downward.

## What was not reset

Full `scontrol show job` before and after the round trip differs only in
`Priority`, `Nice`, `Reason` and `LastSchedEval`:

- `SubmitTime`, `AccrueTime`, `EligibleTime` unchanged (2026-11-11T23:45:41).
  The age term keeps accruing: 66972 -> 67023 across the test.
- `Requeue=0`, `Restarts=0`. No requeue, no restart.
- Resources, node list, `ExcNodeList`, TRES, `Command`, `WorkDir` identical.
- Priority returns exactly to baseline plus age accrual (91228 -> 91245, +17 = the age
  delta). Nothing is lost by parking and reverting.

Derived field that did move: `Reason`.

```
before      : QOSMaxJobsPerUserLimit
Nice=1000   : Priority
Nice=0 back : None
```

`Reason` is recomputed by the scheduler, so this is cosmetic, not a state reset. Other
pending jobs in `acc` show both `QOSMaxJobsPerUserLimit` (16) and `Priority` (6) at the
same time, so both strings occur in this queue.

## Implications for park / promote

1. Promote is only "revert to Nice=0". We cannot go below 0, so no job can outrank the
   natural multifactor priority of a fresh job at nice 0.
2. Parking does not stop a job. Nice only reorders the queue. Once a QOS slot frees
   (`MaxJobsPU=3`), the scheduler starts the highest-priority eligible job. A job at
   priority 1 still starts if it is the only eligible candidate.
3. For a hard gate, use hold, not nice. `scontrol hold` gives `Reason=JobHeldUser` and
   `scontrol release` restores the job. Hold also keeps `SubmitTime`, so age keeps growing.
4. Recipe used here:
   - park: `scontrol update JobId=<id> Nice=250000`
   - promote: `scontrol update JobId=<id> Nice=0`
   - verify with `sprio -l -j <id>` before and after.

## Reproduction

```bash
ssh marenostrum_new
scontrol show job 48941057 | grep -E "Priority=|JobState=|SubmitTime="
sprio -l -j 48941057
scontrol update JobId=48941057 Nice=250000
scontrol show job 48941057 | grep -E "Priority=|JobState="
scontrol update JobId=48941057 Nice=0
```

## Ceiling of this test

- One job on one cluster, one hour, with no preemption and no decay configured. A cluster
  with `PriorityDecayHalfLife > 0` or `PreemptType=preempt/partition_prio` would behave
  differently, because nice also feeds preemption comparisons there.
- The promote direction (negative nice) could not be tested: the site denies it to users.

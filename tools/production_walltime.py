"""Resolve a production allocation's maximum walltime from live Slurm policy."""
import os
import subprocess


def query(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def minutes(value):
    if value in ('', 'UNLIMITED', 'INFINITE', 'NONE', 'N/A'):
        return None
    days, _, clock = value.rpartition('-')
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 1:
        total = parts[0]
    elif len(parts) == 3:
        assert parts[2] == 0, 'Sub-minute policy limit unsupported'
        total = parts[0] * 60 + parts[1]
    else:
        raise ValueError('Unrecognized Slurm walltime: ' + value)
    total += int(days or 0) * 1440
    assert total > 0, 'Production is not permitted by this walltime policy'
    return total


def resolve(account, partition, qos, user=None):
    user = user or os.environ['USER']
    part_raw = query(['scontrol', 'show', 'partition', partition, '-o'])
    part = dict(x.split('=', 1) for x in part_raw.split() if '=' in x)
    assert part['PartitionName'] == partition
    qos_raw = query(['sacctmgr', '-n', '-P', 'show', 'qos', 'format=Name,MaxWall,Flags'])
    policies = {r[0]: r[1:] for line in qos_raw.splitlines() if (r := line.split('|'))}
    assert qos in policies, 'Unknown job QoS'
    # Reject precedence changes rather than applying an incorrect minimum.
    assert 'OverPartQOS' not in policies[qos][1], 'QoS precedence needs explicit support'
    caps = {'partition': minutes(part['MaxTime']), 'job_qos': minutes(policies[qos][0])}
    partition_qos = part.get('QoS', 'N/A')
    if partition_qos not in ('N/A', ''):
        assert partition_qos in policies
        caps['partition_qos'] = minutes(policies[partition_qos][0])
    assoc_raw = query(['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user='+user,
                       'account='+account, 'format=Account,User,Partition,QOS,DefaultQOS,MaxWall'])
    matches = [r for line in assoc_raw.splitlines() if (r := line.split('|'))
               and r[:2] == [account, user] and r[2] in ('', partition)]
    assert matches, 'No matching account/partition association'
    assert any(qos in r[3].split(',') for r in matches), 'QoS not authorized for account'
    for i, row in enumerate(matches):
        caps['association_'+str(i)] = minutes(row[5])
    finite = [v for v in caps.values() if v is not None]
    maximum = min(finite) if finite else None
    walltime = f'{maximum//60:02d}:{maximum%60:02d}:00' if maximum else '0'
    return dict(account=account, partition=partition, qos=qos, user=user,
                minutes=maximum, time=walltime, caps=caps,
                evidence=dict(partition=part_raw, qos=qos_raw, associations=assoc_raw))

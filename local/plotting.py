import argparse
import matplotlib.pyplot as plt

import re
import sys
import numpy as np
import os
import datetime

SKIPPED_LINES = 1
TFLOPS_LABEL = "throughput per GPU (TFLOP/s/GPU)"
ELAPSED_TIME_LABEL="elapsed time per iteration (ms)"

ITER_LINE_RE = re.compile(r".*iteration\s+(\d+)\/")
ITER_SPLIT_RE = re.compile(r'\.\.\.+')
ARG_START_RE = re.compile(r'---+ arguments ---+')
ARG_END_RE = re.compile(r'---+ end of arguments ---+')
# String that is displayed at the start of training:
## 0: [before the start of training step] datetime: 2025-10-21 01:04:37 
# reference for parameter parsing
#  > number of parameters on (tensor, pipeline) model parallel rank (6, 0): 8624807936
PARAM_PARSER = re.compile(r'.* parameters on \(tensor, pipeline\) model parallel rank \((\d+), (\d+)\): (\d+)')

START_TIME_RE = re.compile(r'.*\[before the start of training step\] datetime:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
END_TIME_RE = re.compile(r'.*\[after training is done\] datetime:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
ITER_TS_RE = re.compile(r'^\s*\d+:\s+\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+iteration')

def pprint(*args):
    print("  ", *args)

def read_file(file_path):
    with open(file_path) as f:
        return f.readlines()

def parse_args(lines):
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if ARG_START_RE.search(line):
            start_idx = i
        if ARG_END_RE.search(line):
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        return None
    else:
        lines = [line.strip() for line in lines[start_idx+1:end_idx]]
        arguments = {}
        for line in lines:
            if line:
                try:
                    parts = ITER_SPLIT_RE.split(line.split(":")[-1])
                    key = parts[0].strip()
                    value = parts[1].strip()
                    arguments[key] = value
                except:
                    print(f"Failed to grep arguments from line: \"{line}\"")
        return arguments

def parse_iteration(line):
    splits = line.split("|")
    result = {}
    for split in splits[1:-1]:
        key, value = split.split(":")
        key = key.strip()
        value = float(value.strip())
        result[key] = value
    return result

def get_key(key, througput_dict):
    try:
        return [t[key] for t in througput_dict]
    except KeyError:
        return [-1 for t in througput_dict][0]

def extract_params(lines):
    params = [PARAM_PARSER.match(line).groups() for line in lines if PARAM_PARSER.match(line)]
    params = {f"{t}_{p}": int(n) for t, p, n in params}
    params = sum(params.values())
    return params

def extract_start_end_times(lines):
    start_time = None
    end_time = None
    first_iter_ts = None
    last_iter_ts = None
    for line in lines:
        if start_time is None:
            m = START_TIME_RE.match(line)
            if m:
                start_time = m.group(1)
        if end_time is None:
            m2 = END_TIME_RE.match(line)
            if m2:
                end_time = m2.group(1)
        m3 = ITER_TS_RE.match(line)
        if m3:
            ts = m3.group(1)
            if first_iter_ts is None:
                first_iter_ts = ts
            last_iter_ts = ts
    if start_time is None:
        start_time = first_iter_ts
    if end_time is None:
        end_time = last_iter_ts
    return start_time, end_time

def extract_values(filepath, return_loss_min_max=True, *extra_args):
    lines = read_file(filepath)
    args = parse_args(lines)
    # print(args['profile'])
    if args['profile'] != "True":
        throughput = [parse_iteration(line) for line in lines if ITER_LINE_RE.match(line)]
        throughput = throughput[SKIPPED_LINES:]
    else:
        omit_start = int(args['profile_step_start'])
        omit_end = int(args['profile_step_end'])
        throughput = [parse_iteration(line) for line in lines if ITER_LINE_RE.match(line)]
        throughput = throughput[SKIPPED_LINES:omit_start] + throughput[omit_end:]

    num_model_params = extract_params(lines)
    time_created = os.path.getctime(filepath)
    time_created = datetime.datetime.fromtimestamp(time_created).strftime('%Y-%m-%d %H:%M:%S')
    # Extract training start/end times
    start_time_str, end_time_str = extract_start_end_times(lines)
    elapsed_hours = None
    if start_time_str and end_time_str:
        try:
            fmt = '%Y-%m-%d %H:%M:%S'
            start_dt = datetime.datetime.strptime(start_time_str, fmt)
            end_dt = datetime.datetime.strptime(end_time_str, fmt)
            elapsed_hours = (end_dt - start_dt).total_seconds() / 3600.0
        except Exception:
            elapsed_hours = None

    if not args:
        return None
    WORLD_SIZE = int(args["world_size"])
    seq_len = args["seq_length"]
    batch_size = args["global_batch_size"]
    tgs = [int(seq_len)*int(batch_size) / t[ELAPSED_TIME_LABEL]*1000 / WORLD_SIZE  for t in throughput]
    loss = get_key("lm loss", throughput)
    # make loss to be tuple of starting loss and ending loss, including nans
    loss = np.array(loss)
    loss_has_nan = (loss == -1).any()
    if loss_has_nan:
        loss_start = np.nan
        loss_end = np.nan
    else:
        loss_start = loss[0]
        loss_end = loss[-1]
    tflops = get_key(TFLOPS_LABEL, throughput)
    mem_usages = get_key("mem usages", throughput)
    # check if fsdp-key exists
    fsdp_key = 'use_torch_fsdp2'

    if fsdp_key not in args and 'fsdp' not in args:
        args['fsdp'] = False 
    elif fsdp_key in args:
        args['fsdp'] = args[fsdp_key]
    if return_loss_min_max:
        loss = (loss_start, loss_end)

    
    
    result = {
        "tgs": tgs,
        "tflops": tflops,
        "mem_usages": mem_usages,
        "seq_len": seq_len,
        "micro_batch_size": args['micro_batch_size'],
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "data_parallel_size": args['data_parallel_size'],
        "fsdp": args['fsdp'],
        "precision": args['params_dtype'],
        "fp8": args['fp8'],
        "rope_fusion": args['apply_rope_fusion'],
        "lm loss": loss,
        "optimizer": args['optimizer'],
        "embedding_size": args['max_position_embeddings'],
        "transformer_impl": args['transformer_impl'],
        "tensor_model_parallel_size": args['tensor_model_parallel_size'],
        "pipeline_model_parallel_size": args['pipeline_model_parallel_size'],
        "recompute_granularity": args['recompute_granularity'],
        "num_model_params": num_model_params,
        "timestamp": time_created,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "training_hours": elapsed_hours,
        "log_lines": len(tgs),
        "log_interval": args['log_interval'],
        "data_path": args['data_path'],
        "filename": filepath, 
        }
    
    # add optional args from args
    extra_args = {arg_name: get_key(arg_name, throughput) for arg_name in extra_args }
    result.update(extra_args)
    
    return result, len(tgs)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    files = args.files
    output = args.output
    results = []
    for file in files:
        result, _ = extract_values(file, return_loss_min_max=False)
        plt.plot(result['lm loss'], label=file)
    
    plt.ylim(5, 7)
    plt.xlim(0,175)
    plt.legend()
    plt.savefig(output)
    plt.close()
if __name__ == "__main__":
    main()
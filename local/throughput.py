import re
import sys
import numpy as np
import os
import datetime

SKIPPED_LINES = 2
TFLOPS_LABEL = "throughput per GPU (TFLOP/s/GPU)"
ELAPSED_TIME_LABEL="elapsed time per iteration (ms)"
LEARNING_RATE_LABEL = "learning rate"

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

def get_total_iterations(args):
    if args['train_iters']:
        return int(args['train_iters'])
    elif args['train_samples']:
        total_samples = int(args['train_samples'])
        global_batch_size = int(args['global_batch_size'])
        seq_len = int(args['seq_length'])
        return total_samples // (global_batch_size * seq_len)
    else:
        return -1
    
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
    learning_rate = get_key(LEARNING_RATE_LABEL, throughput)
    # check if fsdp-key exists
    fsdp_key = 'use_torch_fsdp2'

    if fsdp_key not in args and 'fsdp' not in args:
        args['fsdp'] = False 
    elif fsdp_key in args:
        args['fsdp'] = args[fsdp_key]
    if return_loss_min_max:
        loss = (loss_start, loss_end)
        learning_rate = (max(learning_rate), min(learning_rate))

    
    
    result = {
        "tgs": tgs,
        "tflops": tflops,
        "mem_usages": mem_usages,
        "learning_rate": learning_rate,
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
        "time_per_iter": get_key(ELAPSED_TIME_LABEL, throughput),
        "total_iters": get_total_iterations(args),

        }
    
    # add optional args from args
    extra_args = {arg_name: get_key(arg_name, throughput) for arg_name in extra_args if arg_name not in result.keys()}
    result.update(extra_args)
    
    return result, len(tgs)
        
   

def main(argv):
    values, log_lines = extract_values(argv[1])
     ### print relevant args 
    print()
    print(f"File: {values['filename']}")
    for key, value in values.items():
        if key == 'filename':
            continue
        elif key in ['tgs', 'tflops', 'mem_usages']:
            continue
        elif key == 'num_model_params':
            print(f"{key}: {value/1e9:.2f}B", end=", ")
        elif key in ['elapsed_hours', 'training_hours'] and value is not None:
            print(f"{key}: {value:.2f}h", end=", ")
        else:
            if isinstance(value, list):
                continue
            else:
                print(f"{key}: {value}", end=", ")
    
    # Compute total GPU hours required to train the model to completion based on current elapsed time per iteration and total number of training iterations
    # We use divider of 2 as one MI250x GPU has 2 compute dies
    
    # gpuh_to_complete = throughput * np.mean([t[ELAPSED_TIME_LABEL] for t in throughput]) / 1000 / 3600 * get_total_iterations(args) / 2
    mean_gpuh_to_complete = np.mean(values['time_per_iter']) / 1000 / 3600 * values['total_iters'] * (values['world_size'] / 2)
    min_gpuh_to_complete = np.min(values['time_per_iter']) / 1000 / 3600 * values['total_iters'] * (values['world_size'] / 2)
    print(f"\nEstimated GPU hours to complete training based on mean time per iteration: {mean_gpuh_to_complete:.2f}h"
          f"\nEstimated GPU hours to complete training based on best time per iteration: {min_gpuh_to_complete:.2f}h")
    # estimated wall-clock time to complete training
    mean_wall_clock_to_complete = np.mean(values['time_per_iter']) / 1000 / 3600 * values['total_iters']
    print(f"Estimated wall-clock time to complete training based on mean time per iteration: {mean_wall_clock_to_complete:.2f}h")
    # ### print throughput numbers
    row_format = "{:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.6e}"
    header_format = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    header = header_format.format(" ", "TGS", "TFLOPs", "time_per_iter", "lr")
    separator = "-" * len(header)
    if log_lines > 1:
        print("")
        print(f"Skipped first {SKIPPED_LINES} iterations. Averaging over {log_lines} log lines")
        print(separator)
        print(header)
        print(separator)
        print(row_format.format("mean", np.mean(values['tgs']), np.mean(values['tflops']), np.mean(values['time_per_iter']), np.mean(values['learning_rate'])))
        print(row_format.format("std", np.std(values['tgs']), np.std(values['tflops']), np.std(values['time_per_iter']), np.std(values['learning_rate'])))
        print(row_format.format("max", np.max(values['tgs']), np.max(values['tflops']), np.max(values['time_per_iter']), np.max(values['learning_rate'])))
        print(row_format.format("min", np.min(values['tgs']), np.min(values['tflops']), np.min(values['time_per_iter']), np.min(values['learning_rate'])))
    else:
        print(f"Found {log_lines} logs lines, not enough")
    

if __name__ == "__main__":
    main(sys.argv)
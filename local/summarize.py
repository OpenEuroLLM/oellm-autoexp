from throughput import extract_values
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def summarize_list_variables(values):
    for key, _ in values.items():
        if isinstance(values[key], list):
            values[key] = np.median(values[key])
        elif key == "num_model_params":
            values[key] = f"{values[key]/1e9:.1f}B"
    return values

    
def pretty_print_csv(csv):
    header, rows = csv.split("\n")[0], csv.split("\n")[1:]
    print(len(rows))
    
def parse_param_string_as_float(s):
    return float(s.split("B")[0])

def parse_params_as_float(df, params):
    operator = "="
    param_count = -1
    print(params)
    if params[0] == "<" or params[0] == ">":
        operator = params[0]
        params = params[1:]
    if "B" in params:
        param_count = float(params.split("B")[0])
    if operator == "<":
        df = df[df["num_model_params"].apply(lambda x: parse_param_string_as_float(x)) < param_count]
    elif operator == ">":
        df = df[df["num_model_params"].apply(lambda x: parse_param_string_as_float(x)) > param_count]
    elif operator == "=":
        df = df[df["num_model_params"].apply(lambda x: parse_param_string_as_float(x)) == param_count]
    return df

def apply_filters(df, filters):
    # tp = tensor model parallel size
    # fsdp = boolean flag for fsdp
    # model = substring of model name from "filename"
    for f in filters:
        key, value = f.split("=")
        print(key, value)
        if key == "tp":
            df = df[df["tensor_model_parallel_size"] == value]
        elif key == "fsdp":
            df = df[df["fsdp"] == int(value)]
        elif key == "model":
            df = df[df["filename"].str.contains(value)]
        elif key == "params":
            print("key", key, "value", value)
            df = parse_params_as_float(df, value)
        elif key == "fp8":
            df = df[df["fp8"] == value]
    return df


def select_today(df):
    df = df[df['timestamp'].dt.date == pd.Timestamp.today().date()]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory to search files in", default=".")
    ap.add_argument('--pattern', help='glob pattern to search files with', default="**/*.out")
    ap.add_argument('--verbose', help='print errors', action='store_true')
    ap.add_argument('--filters', help='a few quick filters to apply options: tp=N fsdp=N model=NB', nargs='+', default=[])
    ap.add_argument('--columns', help='columns to select', nargs='+', default=[])
    ap.add_argument('--today', help='only show today', action='store_true')
    ap.add_argument('--short', help='pretty print csv', action='store_true')
    ap.add_argument('--csv', help='output table that can be copied to sheets', action='store_true')
    ap.add_argument('--tgs_sort', action='store_true', default=False, help='sort by tgs. default is by timestamp')
    args = ap.parse_args()
    files = Path(args.dir).glob(args.pattern)
    files = [f for f in files]
    print(f"Found {len(files)} files")
    result = []
    for file in files:
        try:
            values, log_lines  = extract_values(file)
            if values['tgs'] == []:
                continue
            else:
                values = summarize_list_variables(values)
                values["log_lines"] = log_lines
                result.append(values)
            assert False
        except Exception as e:
            if args.verbose:
                print(f"Error: {e}")
            continue
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    df = pd.DataFrame(result)
    df['timestamp'] = df['timestamp'].apply(lambda x: pd.Timestamp(x))
    df['filename'] = df['filename'].apply(lambda x: str(x))
    df = apply_filters(df, args.filters)
    if args.today:
        df = select_today(df)
    if args.columns:
        df = df[args.columns]
    # sort by timestamp
    if args.tgs_sort:
        df = df.sort_values(by="tgs", ascending=False)
    else:
        df = df.sort_values(by="timestamp")
    # delete index column
    df = df.reset_index(drop=True)
    try:
        df = df.drop(['dpa_args'], axis=1) 
    except:
        pass
    if args.short:
        df = df.drop(['transformer_impl'], axis=1)
        df['precision'] = df['precision'].str.replace("torch.", "")
        # cut long column names to first characters
        def shorten_col(col):
            s = col.split("_")
            return ".".join([c[:1] for c in s])
        df.columns = [shorten_col(col) if len(col)>10 else col for col in df.columns]
        # drop two last cols
        df = df.iloc[:, :-2]
    if args.csv:
        print(df.to_csv())
    else:
        print(df.to_string())

main()
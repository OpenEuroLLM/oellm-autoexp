import sys
from throughput import parse_args

def read_lines(path):
    with open(path) as f:
        return f.readlines()

def main(argv):
    values1 = parse_args(read_lines(argv[1]))
    values2 = parse_args(read_lines(argv[2]))
    keys1, keys2 = set(values1.keys()), set(values2.keys())
    
    # Print differences in keys
    for diff in keys1.difference(keys2):
        print(f"++ {diff}")
    for diff in keys2.difference(keys1):
        print(f"-- {diff}")

    # Calculate the longest key length for alignment
    shared_keys = keys1.intersection(keys2)
    max_key_len = max(len(key) for key in shared_keys)

    # Print shared keys with values properly aligned
    for key in shared_keys:
        val1, val2 = values1[key], values2[key]
        if val1 != val2:
            print(f"{key:<{max_key_len}}  {val1}  {val2}")

if __name__ == "__main__":
    main(sys.argv)

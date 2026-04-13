import json

file1 = json.load(open('compare_json/ville.json', 'r'))
file2 = json.load(open('compare_json/niccolo.json', 'r'))

def compare_dicts(d1, d2, path=""):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    # Collect results
    missing_keys_in_d1 = [f"Missing from the second file: '{path + key}" for key in keys1 - keys2]
    missing_keys_in_d2 = [f"Missing from the first file:  '{path + key}'" for key in keys2 - keys1]
    differing_values = {}

    # Process common keys
    common_keys = keys1 & keys2
    for key in common_keys:
        val1 = d1[key]
        val2 = d2[key]
        
        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recursively compare nested dictionaries
            nested_missing_keys_in_d1, nested_missing_keys_in_d2, nested_differing_values = compare_dicts(val1, val2, path + key + ".")
            missing_keys_in_d1.extend(nested_missing_keys_in_d1)
            missing_keys_in_d2.extend(nested_missing_keys_in_d2)
            differing_values.update(nested_differing_values)
        elif val1 != val2:
            differing_values[path + key] = (val1, val2)

    # Print results in the desired order
    if path == "":  # Only print at the top level
        for msg in missing_keys_in_d1:
            print(msg, flush=True)
        for msg in missing_keys_in_d2:
            print(msg, flush=True)
        for key, (val1, val2) in differing_values.items():
            print(f"Different values for key: '{key}':\n"
                  f"    First file:  {val1}\n"
                  f"    Second file: {val2}", flush=True)

    return missing_keys_in_d1, missing_keys_in_d2, differing_values

if __name__ == "__main__":
    compare_dicts(file1, file2)
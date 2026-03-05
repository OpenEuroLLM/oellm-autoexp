import regex as re
import sys

# Example line to grep:
# [2025-12-16 14:44:49] iteration      265/   14306 | consumed samples:       271360 | elapsed time per iteration (ms): 3769.8 | throughput per GPU (TFLOP/s/GPU): 36.9 | learning rate: 5.559441E-05 | global batch size:  1024 | lm loss: 5.941908E+00 | load_balancing_loss: 1.026827E+00 | loss scale: 1.0 | grad norm: 2.215 | number of skipped iterations:   0 | number of nan iterations:   0 |
# Grep elapsed time per iteration (ms):

pattern = r"elapsed time per iteration \(ms\): ([0-9]+\.[0-9]+)"
sec_in_hour=3600

# CHANGE THIS based on your run
total_tokens=30e9
gbs_in_tokens=(2048*1024)
world_size=4*8

def main(sys_argv):
    if len(sys_argv) != 2:
        print("Usage: python compute_gpuh.py <log_file>")
        sys.exit(1)

    log_file = sys_argv[1]
    with open(log_file, 'r') as f:
        log_data = f.read()

    matches = re.findall(pattern, log_data)
    if not matches:
        print("No matches found for elapsed time per iteration.")
        sys.exit(1)

    elapsed_times = [float(match) for match in matches]
    avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
    print(f"Average elapsed time per iteration (ms): {avg_elapsed_time}")

    elapsed_time_per_iter=avg_elapsed_time/1000.0
    est_training_hours = total_tokens/((gbs_in_tokens/elapsed_time_per_iter)*sec_in_hour)
    num_lumi_gpus = world_size / 2
    print("Estimated Training Time (hours): ", est_training_hours)
    print("Estimated GPUh burned (GPUh): ", est_training_hours*num_lumi_gpus)
if __name__ == "__main__":
    main(sys.argv)
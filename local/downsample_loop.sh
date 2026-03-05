#!/bin/bash

for file in /scratch/project_462000963/preprocessed/gpt-neox-20b/nemotron-cc/1.0/*bin; do 
    skip_files=(
        "high-all"
    )
    # do not run for "high-actual" or "high-all"
    if [[ " ${skip_files[@]} " =~ " $(basename "$file" .bin) " ]]; then
        echo "Skipping file: $file"
        continue
    fi
    file_prefix=${file%.bin}
    output_dir="/flash/project_462000353/rluukkon/downsampled_data/$(basename "$file_prefix")"
    # mkdir -p "$output_dir"
    echo "Processing file: $file_prefix, outputting to: $output_dir"
    # if $1 == dry-run, then do not submit the job
    if [ "$1" == "dry-run" ]; then
        echo "Dry run: sbatch local/downsample.slurm  \"$file_prefix\" \"$output_dir\""
        continue
    fi
    sbatch local/downsample.slurm  "$file_prefix" "$output_dir"
done
# done
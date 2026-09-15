#!/bin/bash

csv_file="/share/data_share/UGV_lidar_data/PlotSplits_NYH2_NYH3_2020-24.csv"
file_dir="/share/data_share/UGV_lidar_data/2021/2021_NYH2_projected_lidar_denoised"
destination_dir="/share/data_share/UGV_lidar_data/CV_Experiments/lidar_NYH2_NYH3_2020-24_projected_lidar_denoised"

# Substrings to exclude (not full filename matches)
exclude_substrings=("range_1_no_flip.png" "range_1.png" "range_12_no_flip.png" "range_12.png")

# Extract collection IDs from column 3
ids=$(awk -F',' 'NR>1 {print $2}' "$csv_file")

cd "$file_dir" || exit 1

for id in $ids; do
  echo "Searching for files containing: $id"

  for file in *"$id"*; do
    [ -e "$file" ] || continue

    # Skip file if it contains any excluded substring
    skip=false
    for pattern in "${exclude_substrings[@]}"; do
      if [[ "$file" == *"$pattern"* ]]; then
        skip=true
        break
      fi
    done

    # Skip file if already exists in destination
    if [ -e "$destination_dir/$file" ]; then
      echo "Skipping (already exists): $file"
      continue
    fi

    # Copy only if not excluded and doesn't already exist
    if [ "$skip" = false ]; then
      echo "Copying: $file"
      cp -v "$file" "$destination_dir"
    fi
  done
done

#manually moved range 1 and removed range 11 of columns 3 & 4 in 2021

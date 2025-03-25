#!/bin/bash

# Define target options
dataset="ML_1M" #ML_100K, ML_1M or STEAM
seed=42

# Dataset specific options
# MovieLens
target_cat_options_ml=("Horror" "Action" "Adventure" "Animation" "Fantasy" "Drama")
# MovieLens 1M
target_items_options_ml1m=(2858 2005 728 2738)
num_users_ml1m=None
sample_num_ml1m=200

# MovieLens 100K
target_items_options_ml100k=(50 411 630 1305)
num_users_ml100k=None
sample_num_ml100k=None

# STEAM
target_cat_options_steam=("Action" "Indie" "Free to Play" "Sports" "Photo Editing")
target_items_options_steam=(271590 35140 292140 582160)
num_users_steam=None
sample_num_steam=200


# Check if sufficient arguments were provided
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <start_index> <end_index> <model> <target_mode> [categorized]"
    echo "target_mode: targeted | untargeted"
    echo "categorized: categorized | uncategorized (optional, defaults to uncategorized)"
    exit 1
fi

# Parse arguments
start=$1
end=$2
model=$3
target_mode=$4
categorized=${5:-uncategorized}  # Default to uncategorized if not provided

shift 5  # Shift arguments so we can process additional flags

# Parse optional flags
# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed=*)
            seed="${1#*=}"  # Extract seed value
            ;;
        --dataset=*)
            dataset="${1#*=}"  # Extract dataset value
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# Set target_items_options based on dataset
if [[ "$dataset" == "ML_1M" ]]; then
    target_cat_options=("${target_cat_options_ml[@]}")
    target_items_options=("${target_items_options_ml1m[@]}")
    num_users=("${num_users_ml1m[@]}")
    sample_num=("${sample_num_ml1m[@]}")
elif [[ "$dataset" == "ML_100K" ]]; then
    target_cat_options=("${target_cat_options_ml[@]}")
    target_items_options=("${target_items_options_ml100k[@]}")
    num_users=("${num_users_ml100k[@]}")
    sample_num=("${sample_num_ml100k[@]}")
elif [[ "$dataset" == "STEAM" ]]; then
    target_cat_options=("${target_cat_options_steam[@]}")
    target_items_options=("${target_items_options_steam[@]}")
    num_users=("${num_users_steam[@]}")
    sample_num=("${sample_num_steam[@]}")
else
    echo "Error: Invalid dataset. Choose 'ML_100K', 'ML_1M' and 'STEAM'"
    exit 1
fi

# Determine total iterations based on mode
if [[ "$target_mode" == "targeted" && "$categorized" == "uncategorized" ]]; then
    total_iterations=${#target_items_options[@]}
elif [[ "$target_mode" == "targeted" && "$categorized" == "categorized" ]]; then
    total_iterations=${#target_cat_options[@]}
else
    total_iterations=1
fi

# Validate indices
if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
    echo "Error: Both <start_index> and <end_index> must be integers."
    exit 1
fi

if (( start < 1 || end < 1 || start > total_iterations || end > total_iterations || start > end )); then
    echo "Error: Invalid range. Ensure 1 <= start_index <= end_index <= $total_iterations."
    exit 1
fi

# Adjust indices to be zero-based for the loop
start=$((start - 1))
end=$((end - 1))

# Determine target options based on mode
if [[ "$target_mode" == "targeted" && "$categorized" == "uncategorized" ]]; then
    target_options=("${target_items_options[@]}")
elif [[ "$target_mode" == "targeted" && "$categorized" == "categorized" ]]; then
    target_options=("${target_cat_options[@]}")
elif [[ "$target_mode" == "untargeted" ]]; then
    target_options=("") # No target option needed
else
    echo "Error: Invalid mode combination."
    exit 1
fi

# Initialize iteration counter
iteration=0

# Iterate over target options
for target in "${target_options[@]}"; do
    ((iteration++))

    # Skip iterations outside the specified range
    if (( iteration < start + 1 || iteration > end + 1 )); then
        continue
    fi

    echo "Iteration $iteration of $total_iterations (Executing range $((start + 1)) to $((end + 1)))"

    # Construct JSON configuration
    config_json=$(cat <<EOF
{
"settings": {
    "model": "$model",
    "device": "cpu",
    "dataset": "$dataset",
    "seed": $seed
    },
    $( [[ "$target_mode" == "targeted" ]] && echo "\"evolution\": { \"target_cat\": $target }," )
  "generation": {
    "targeted": $( [[ "$target_mode" == "targeted" ]] && echo "True" || echo "False" ),
    "categorized": $( [[ "$categorized" == "categorized" ]] && echo "True" || echo "False" )
  }
}
EOF
)

    # Print the configuration for debugging
echo "========EVALUATION.SH INFO========"
echo "Running script:"
echo "python -m bin.cli evaluate alignment \\
    --use-cache=False \\
    --save-path=\"results/evaluate/alignment.db\" \\
    --config_dict='$config_json' \\
    --mode=\"all\" \\
    --range-i=\"(0, $num_users)\" \\
    --splits=\"[(None, 10, 0)]\" \\
    --sample_num=$sample_num
    "
echo "=================================="

   # Run the script
   python -m bin.cli evaluate alignment \
       --use-cache=False \
       --save-path="results/evaluate/alignment.db" \
       --config_dict="$config_json" \
       --mode="all" \
       --range-i="(0, $num_users)" \
       --splits="[(None, 10, 0)]" \
       --sample_num=$sample_num


done

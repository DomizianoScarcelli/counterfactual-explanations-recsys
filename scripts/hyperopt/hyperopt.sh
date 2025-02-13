#!/bin/bash


target_cat_options=("Horror" "Action" "Adventure" "Animation" "Fantasy" "Drama")

# Calculate the total number of iterations
total_iterations=$(( ${#target_cat_options[@]}))

# Check if arguments were provided
start=$1
end=$2
model=$3

if [[ -z "$start" || -z "$end" ]]; then
    echo "Usage: $0 <start_index> <end_index>"
    echo "Both <start_index> and <end_index> must be integers between 1 and $total_iterations."
    exit 1
fi

# Validate that start and end are integers
if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
    echo "Error: Both <start_index> and <end_index> must be integers."
    exit 1
fi

# Validate the range
if (( start < 1 || end < 1 || start > total_iterations || end > total_iterations || start > end )); then
    echo "Error: Invalid range. Ensure 1 <= start_index <= end_index <= $total_iterations."
    exit 1
fi

# Adjust indices to be zero-based for the loop
start=$((start - 1))
end=$((end - 1))

# Initialize iteration counter
iteration=0

# Iterate over all combinations of parameters
for target_cat in "${target_cat_options[@]}"; do
    # Increment the iteration counter
    ((iteration++))

    # Skip iterations outside the specified range
    if (( iteration < start + 1 || iteration > end + 1 )); then
        continue
    fi

    # Print progress
    echo "Iteration $iteration of $total_iterations (Executing range $((start + 1)) to $((end + 1)))"

    # Define the JSON configuration in a variable
    config_json=$(cat <<EOF
{
"settings": {
    "model": $model,
    "device": "cpu"
    },
  "evolution": {
    "target_cat": $target_cat,
  },
}
EOF
)

                                # Print the configuration being tested (for debugging)
                                echo "Running script with configuration:"
                                echo "$config_json"

                                # Run the script with the JSON string as the --config-dict argument
                                python -m cli evaluate alignment \
                                    --use-cache=False \
                                    --save-path="results/evaluate/alignment/alignment.db" \
                                    --config_dict="$config_json" \
                                    --mode="all" \
                                    --range-i="(0, None)" \
                                    --splits="[(None, 10, 0)]" \
                                    --target-cat=$target_cat
                                done

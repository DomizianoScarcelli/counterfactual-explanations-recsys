#!/bin/bash

categorized_options=("False" "True")

# Calculate the total number of iterations
total_iterations=2

# Check if arguments were provided
start=$1
end=$2

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
for categorized in "${categorized_options[@]}"; do
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
    "model": "BERT4Rec",
    "device": "cpu"
    },
"generation": {
    "targeted": False,
    "categorized": $categorized 
    }
}

EOF
)

    # Print the configuration being tested (for debugging)
    echo "Running script with configuration:"
    echo "$config_json"

    # Run the script with the JSON string as the --config-dict argument
    python -m cli evaluate sensitivity \
        --save-path="results/evaluate/sensitivity/sensitivity_untargeted.csv" \
        --config_dict="$config_json" \
        --target-cat=$target_cat
    done

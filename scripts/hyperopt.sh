#!/bin/bash

# Define the possible values for each parameter
crossover_prob_options=(0.7)
mutation_prob_options=(0.5)
fitness_alpha_options=(0.5 0.7)
generations_options=(10 20 30)
pop_size_options=(2048 4096 8192 16384)
similarity_threshold_options=(0.7 0.5)

# Calculate the total number of iterations
total_iterations=$(( ${#crossover_prob_options[@]} * ${#mutation_prob_options[@]} * ${#fitness_alpha_options[@]} * ${#generations_options[@]} * ${#pop_size_options[@]} * ${#similarity_threshold_options[@]} ))

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
for crossover_prob in "${crossover_prob_options[@]}"; do
    for mutation_prob in "${mutation_prob_options[@]}"; do
        for fitness_alpha in "${fitness_alpha_options[@]}"; do
            for generations in "${generations_options[@]}"; do
                for pop_size in "${pop_size_options[@]}"; do
                    for similarity_threshold in "${similarity_threshold_options[@]}"; do
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
  "evolution": {
    "crossover_prob": $crossover_prob,
    "fitness_alpha": $fitness_alpha,
    "generations": $generations,
    "mut_prob": $mutation_prob,
    "pop_size": $pop_size
  },
  "generation": {
    "similarity_threshold": $similarity_threshold
  }
}
EOF
)

                        # Print the configuration being tested (for debugging)
                        echo "Running script with configuration:"
                        echo "$config_json"

                        # Run the script with the JSON string as the --config-dict argument
                        python -m cli evaluate alignment \
                            --use-cache=False \
                            --save-path="results/evaluate/alignment/alignment_hyperopt.csv" \
                            --config_dict="$config_json" \
                            --mode="all" \
                            --range-i="(0, 100)" \
                            --splits="[(None, 5, 0), (None, 10, 0), (None, 5, 5), (None, 10, 5)]"
                    done
                done
            done
        done
    done
done

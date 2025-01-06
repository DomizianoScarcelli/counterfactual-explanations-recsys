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

# Check if an argument was provided
filter=$1  # Accept the first argument as the filter
if [[ -z "$filter" ]]; then
    filter="all"
elif [[ "$filter" != "odd" && "$filter" != "even" ]]; then
    echo "Invalid argument: $filter. Use 'odd', 'even', or leave empty for all iterations."
    exit 1
fi

# Determine the behavior based on the filter
is_odd=-1  # Default: run all iterations
if [[ "$filter" == "odd" ]]; then
    is_odd=1
elif [[ "$filter" == "even" ]]; then
    is_odd=0
fi

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

                        # Skip iterations based on the filter
                        if [[ "$is_odd" -ne -1 && $((iteration % 2)) -ne "$is_odd" ]]; then
                            continue
                        fi

                        # Print progress
                        echo "Iteration $iteration of $total_iterations (Executing $filter iterations)"

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
                            --save-path="results/evaluate/alignment/alignment_hyperopt_$filter.csv" \
                            --config_dict="$config_json" \
                            --mode="all" \
                            --range-i="(0, 200)" \
                            --splits="[(None, 5, 0), (None, 10, 0), (None, 5, 5), (None, 10, 5)]"
                    done
                done
            done
        done
    done
done

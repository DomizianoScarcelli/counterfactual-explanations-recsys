#!/bin/bash

#
# This script performs an exhaustive grid search over multiple parameter combinations
# to evaluate the performance of the counterfactual generation algorithm. 
#
# Parameters:
# - `crossover_prob_options`: Array of possible values for crossover probability.
# - `mutation_prob_options`: Array of possible values for mutation probability.
# - `fitness_alpha_options`: Array of possible values for fitness alpha.
# - `generations_options`: Array of possible values for the number of generations.
# - `pop_size_options`: Array of possible values for population size.
# - `similarity_threshold_options`: Array of possible values for similarity threshold.
#
# The script calculates the total number of iterations and systematically iterates
# through all possible parameter combinations. For each combination, it:
# - Generates a JSON configuration with the chosen parameter values.
# - Calls a Python script (`performance_evaluation.automata_learning.evaluate`) with
#   the generated configuration as an argument.
#

# Define the possible values for each parameter
# crossover_prob_options=(0.2 0.5 0.7)
# mutation_prob_options=(0.2 0.5 0.7)
# fitness_alpha_options=(0.2 0.5 0.7)
crossover_prob_options=(0.7)
mutation_prob_options=(0.5)
fitness_alpha_options=(0.5 0.7)
generations_options=(10 20 30)
pop_size_options=(2048 4096 8192 16384)
similarity_threshold_options=(0.5)

# Calculate the total number of iterations
total_iterations=$(( ${#crossover_prob_options[@]} * ${#mutation_prob_options[@]} * ${#fitness_alpha_options[@]} * ${#generations_options[@]} * ${#pop_size_options[@]} * ${#similarity_threshold_options[@]} ))

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

                        # Print progress
                        echo "Iteration $iteration of $total_iterations"

                        # Directly define the JSON configuration in a variable
                        config_json=$(cat <<EOF
{
  "evolution": {
    "crossover_prob": $crossover_prob,
    "fitness_alpha": $fitness_alpha,
    "generations": $generations,
    "mut_prob": $mutation_prob,
    "pop_size": $pop_size,
  },
  "generation": {
    "similarity_threshold": $similarity_threshold
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
                            --save-path="results/evaluate/alignment/alignment_hyperopt_all_with_splits.csv" \
                            --config_dict="$config_json" \
                            --mode="all" \
                            --range-i="(0, 20)" \
                            --splits="[(None, 1, 0), (None, 10, 0), (None, 1, 5), (None, 5, 5), (None, 10, 5)]"
                    done
                done
            done
        done
    done
done

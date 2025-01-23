#!/bin/bash

# Define the possible values for each parameter
crossover_prob_options=(0.7) #0.7 is better than 0.2
mutation_prob_options=(0.5) #0.5 is better than 0.2
fitness_alpha_options=(0.5) #0.7 is better than 0.5 and 0.85
generations_options=(10) #20 and 10 are the best
# pop_size_options=(2048 4096 8192 16384)
pop_size_options=(8192) #2048 and 8192 seems better than other values. 8192 being better than all others.
similarity_threshold_options=(0.5) #0.5 seems better than 0.7
genetic_topk_options=(1) # 1 is better than 5
num_mutations_options=(1) #1 seems better than other ones
target_cat_options=("Action" "Adventure" "Animation" "Horror" "Fantasy")

# Calculate the total number of iterations
total_iterations=$(( ${#crossover_prob_options[@]} * ${#mutation_prob_options[@]} * ${#fitness_alpha_options[@]} * ${#generations_options[@]} * ${#pop_size_options[@]} * ${#similarity_threshold_options[@]} * ${#genetic_topk_options[@]} * ${#num_mutations_options[@]} ^ ${#target_cat_options[@]} ))

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
                    for genetic_topk in "${genetic_topk_options[@]}"; do
                        for similarity_threshold in "${similarity_threshold_options[@]}"; do
                            for num_mutations in "${num_mutations_options[@]}"; do
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
  "evolution": {
    "crossover_prob": $crossover_prob,
    "fitness_alpha": $fitness_alpha,
    "generations": $generations,
    "mut_prob": $mutation_prob,
    "pop_size": $pop_size,
    "mutations": {
        "num_additions": $num_mutations,
        "num_deletions": $num_mutations,
        "num_replaces": $num_mutations,
},
    "target_cat": $target_cat,
  },
  "generation": {
    "similarity_threshold": $similarity_threshold,
    "ignore_genetic_split": True,
    "genetic_topk": $genetic_topk
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
                                    --range-i="(0, 400)" \
                                    --splits="[(None, 10, 0)]" \
                                    --target-cat=$target_cat
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

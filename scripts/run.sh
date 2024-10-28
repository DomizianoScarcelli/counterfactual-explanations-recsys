#!/bin/bash

# Loop from 1 to 10 to run the command with each mutation type
for i in {1..10}
do
  python -m performance_evaluation.alignment.evaluate --mode evaluate --split-type="${i}_mut"
done

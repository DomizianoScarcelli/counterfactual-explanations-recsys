# Demystifying Sequential Recommendations: Counterfactual Explanations via Genetic Algorithms and Automata Learning

## Installation
It's recommended to use `conda` as virtual environment for python. To create and activate the environment with all the requirements, execute the command
``` 
conda env create -f environment.yaml
conda activate recsys
```

## Documentation

The two methods GENE and PACE are evaluated both in the same run. To run an evaluation, use:

```
bash scripts/evaluation.sh <start_index> <end_index> <model> <target_mode> [categorized] [--seed]
```
where

```
model: BERT4Rec | SASRec | GRU4Rec
target_mode: targeted | untargeted
categorized: categorized | uncategorized (optional, defaults to uncategorized)
```

`start_index` and `end_index` determine which range of targets to perform the evaluation on. If untargeted, then `start_index` and `end_index` must be 1.


- `bash scripts/evaluation.sh 1 1 BERT4Rec untargeted --seed=10` runs the untargeted, uncategorized using the BERT4Rec model
- `bash scripts/evaluation.sh 1 6 SASRec untargeted categorized` runs the untargeted, categorized using the SASRec model
- `bash scripts/evaluation.sh 1 6 BERT4Rec targeted categorized` runs the targeted, categorized on all the 6 chosen categories. It uses the standard seed (42)
- `bash scripts/evaluation.sh 1 4 BERT4Rec targeted uncategorized` runs the targeted, categorized on all the 6 chosen categories. It uses the standard seed (42)


All evaluation results are stored into the sqlite3 database at `results/evaluate/alignment.db`, which is created if it doesn't already exists. Already performed runs will be skipped.

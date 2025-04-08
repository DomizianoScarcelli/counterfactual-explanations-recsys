# Demystifying Sequential Recommendations: Counterfactual Explanations via Genetic Algorithms and Automata Learning

## Installation
It's recommended to use `conda` as virtual environment for python. To create and activate the environment with all the requirements, execute the command
``` 
conda env create -f environment.yaml
conda activate recsys
```

## Documentation

### Method Evaluation
To run an evaluation, use:

```
bash scripts/evaluation.sh <start_index> <end_index> <model> <target_mode> [categorized] [--seed]
```
where

```
model: BERT4Rec | SASRec | GRU4Rec
target_mode: targeted | untargeted
categorized: categorized | uncategorized (optional, defaults to uncategorized)
dataset: ML_100K | ML_1M | STEAM
```

`start_index` and `end_index` determine which range of targets to perform the evaluation on. If untargeted, then `start_index` and `end_index` must be 1.


- `bash scripts/evaluation.sh 1 1 BERT4Rec untargeted --seed=10` runs the untargeted, uncategorized using the BERT4Rec model
- `bash scripts/evaluation.sh 1 6 SASRec untargeted categorized` runs the untargeted, categorized using the SASRec model
- `bash scripts/evaluation.sh 1 6 BERT4Rec targeted categorized` runs the targeted, categorized on all the 6 chosen categories. It uses the standard seed (42)
- `bash scripts/evaluation.sh 1 4 BERT4Rec targeted uncategorized` runs the targeted, categorized on all the 6 chosen categories. It uses the standard seed (42)


All evaluation results are stored into the sqlite3 database at `results/evaluate/alignment.db`, which is created if it doesn't already exists. Already performed runs will be skipped.

### Baselines Evaluation
To run the baseline evalution, use:

```
python -m bin.cli utils baselines --dataset --baseline --num-samples --seed --model --num-edits
```
where

```
dataset: ml-100k | ml-1m | steam
model: BERT4Rec | SASRec | GRU4Rec
baseline: random (replace the item with a random item) | educated (replace the item with the target item or an item in the same category)
num-samples: (int) int defaults to the dataset's number of users, refers to the randomly sampled users to run the baseline on.
seed: (int) defaults to 42
num-edits: (int) defaults to 1, refers to the number of edits to allow for each counterfactual generation.
```

Examples:
```
python -m bin.cli utils baselines --dataset="ml-100k" --baseline="educated"  --model="GRU4Rec" --num-edits=2
python -m bin.cli utils baselines --dataset="steam" --baseline="random" --num-samples=200 --seed=10 --model="BERT4Rec" --num-edits=2
python -m bin.cli utils baselines --dataset="steam" --baseline="random" --num-samples=200 --seed=101 --model="GRU4Rec" --num-edits=1
```

All evaluation results are stored into the sqlite3 database at `results/evaluate/baselines.db`, which is created if it doesn't already exists. Already performed runs will be skipped.



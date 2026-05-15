# Demystifying Sequential Recommendations: Counterfactual Explanations via Genetic Algorithms and Automata Learning

[![arXiv](https://img.shields.io/badge/arXiv-2508.03606v1-b31b1b.svg)](https://arxiv.org/pdf/2508.03606v1)

> **Paper Accepted to the main track of the XAI-2026 conference**
> 
> A preprint of this work is available on [arXiv](https://arxiv.org/pdf/2508.03606v1). 
> 
> *Note: The current preprint focuses on the counterfactual generation and evaluation aspects of the project. The automata learning components are not included.*
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

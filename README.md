# TRACE - Trace-based Automata-driven Counterfactual Examples

## Installation

## Documentation

### Generate counterfactual
Run the generation of counterfactuale examples on all (or a part) of the test set

```
python -m cli run [flags]
```

### Evaluation

```
python -m cli evaluate [alignment | automata_learning | generation | sensitivity] [flags]
```

### Evaluate alignment
```
python -m cli evaluate alignment --splits="[(None, 1, 0), (None, 10, 0), (None, 1, 5), (None, 5, 5), (None, 10, 5)]" --use-cache=False --range_i="(0, 50)" --save_path="results/alignment_evaluation.csv"
```

### Evaluate model sensitivity

```
python -m cli evaluate sensitivity --k=10 --target=category  --log-path="results/sensitivity_evaluation.csv"
```

### Evaluate automata learning
```
TODO
```

### Evaluate dataset generation
```
TODO
```
             

### Statistics

```
python -m cli stats [alignment | automata_learning | generation | sensitivity] [flags]
```

## Directory Structure


from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Raw data from the user
raw_data = """Drama BERT4Rec ml-100k GENE 1.0 0.648 0.534 0.42 1.002 1.0 1.0 1.0
Drama BERT4Rec ml-100k PACE 0.585 0.469 0.405 0.327 0.4 0.412 0.401 0.442
Drama GRU4Rec ml-100k GENE 1.0 0.603 0.488 0.397 1.006 1.007 1.007 1.0
Drama GRU4Rec ml-100k PACE 0.512 0.426 0.383 0.319 0.402 0.4 0.41 0.402
Drama SASRec ml-100k GENE 1.0 0.715 0.63 0.552 1.0 1.0 1.0 1.0
Drama SASRec ml-100k PACE 0.595 0.512 0.462 0.435 0.345 0.356 0.335 0.333
Action BERT4Rec ml-100k GENE 1.0 0.758 0.632 0.523 1.0 1.0 1.0 1.0
Action BERT4Rec ml-100k PACE 0.476 0.467 0.416 0.372 0.41 0.443 0.413 0.407
Action GRU4Rec ml-100k GENE 1.0 0.789 0.657 0.547 1.007 1.003 1.006 1.002
Action GRU4Rec ml-100k PACE 0.375 0.411 0.401 0.365 0.404 0.397 0.397 0.384
Action SASRec ml-100k GENE 1.0 0.776 0.644 0.525 1.0 1.0 1.0 1.0
Action SASRec ml-100k PACE 0.403 0.439 0.39 0.359 0.413 0.386 0.378 0.383
Adventure BERT4Rec ml-100k GENE 0.997 0.594 0.353 0.209 1.023 1.016 1.012 1.02
Adventure BERT4Rec ml-100k PACE 0.294 0.233 0.159 0.097 0.491 0.518 0.487 0.484
Adventure GRU4Rec ml-100k GENE 1.0 0.628 0.35 0.166 1.033 1.015 1.003 1.0
Adventure GRU4Rec ml-100k PACE 0.238 0.225 0.139 0.084 0.536 0.557 0.565 0.57
Adventure SASRec ml-100k GENE 0.999 0.626 0.403 0.216 1.008 1.005 1.0 1.0
Adventure SASRec ml-100k PACE 0.268 0.22 0.163 0.098 0.573 0.551 0.526 0.457
Horror BERT4Rec ml-100k GENE 0.993 0.358 0.191 0.139 1.077 1.08 1.072 1.046
Horror BERT4Rec ml-100k PACE 0.176 0.081 0.052 0.036 0.735 0.711 0.633 0.529
Horror GRU4Rec ml-100k GENE 0.994 0.163 0.029 0.012 1.2 1.11 1.0 1.0
Horror GRU4Rec ml-100k PACE 0.103 0.028 0.012 0.012 0.608 0.615 0.545 0.455
Horror SASRec ml-100k GENE 0.995 0.232 0.094 0.054 1.097 1.041 1.022 1.0
Horror SASRec ml-100k PACE 0.151 0.046 0.032 0.023 0.683 0.395 0.267 0.182
Animation BERT4Rec ml-100k GENE 0.943 0.432 0.245 0.158 1.143 1.133 1.108 1.107
Animation BERT4Rec ml-100k PACE 0.146 0.088 0.063 0.037 0.87 0.892 0.898 0.943
Animation GRU4Rec ml-100k GENE 0.981 0.238 0.07 0.016 1.319 1.263 1.167 1.267
Animation GRU4Rec ml-100k PACE 0.099 0.036 0.012 0.005 0.806 0.706 0.545 0.4
Animation SASRec ml-100k GENE 0.994 0.357 0.188 0.115 1.113 1.086 1.102 1.102
Animation SASRec ml-100k PACE 0.154 0.069 0.046 0.027 0.855 0.923 0.977 0.96
Fantasy BERT4Rec ml-100k GENE 0.893 0.256 0.018 0.0 1.401 1.382 1.235 nan
Fantasy BERT4Rec ml-100k PACE 0.128 0.037 0.0 0.0 0.851 0.914 nan nan
Fantasy GRU4Rec ml-100k GENE 0.913 0.317 0.018 0.001 1.628 1.612 1.706 1.0
Fantasy GRU4Rec ml-100k PACE 0.041 0.019 0.002 0.0 0.769 0.778 1.0 nan
Fantasy SASRec ml-100k GENE 0.931 0.354 0.029 0.0 1.34 1.34 1.04 nan
Fantasy SASRec ml-100k PACE 0.141 0.05 0.002 0.0 0.894 0.864 1.0 nan"""

# Process the data
model_fidelity_1 = defaultdict(list)
model_fidelity_5 = defaultdict(list)
targets = []

for line in raw_data.split("\n"):
    parts = line.split()
    target = parts[0]
    fidelity_1, fidelity_5 = float(parts[4]), float(parts[5])
    model_fidelity_1[target].append(fidelity_1)
    model_fidelity_5[target].append(fidelity_5)
    if target not in targets:
        targets.append(target)

# Compute averages
avg_fidelity_1 = [np.mean(model_fidelity_1[t]) for t in targets]
avg_fidelity_5 = [np.mean(model_fidelity_5[t]) for t in targets]

# Plot histogram
x = np.arange(len(targets))
width = 0.4

plt.figure(figsize=(6, 6))
plt.bar(x - width/2, avg_fidelity_1, width, label="Fidelity@1", color='#007edc')
plt.bar(x + width/2, avg_fidelity_5, width, label="Fidelity@5", color='#dd1655')
plt.xlabel("Target")
plt.ylabel("Average Fidelity")
plt.title("Targeted-Categorized: Average Fidelity@1 and Fidelity@5 per Target")
plt.xticks(x, targets, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig("target.png")


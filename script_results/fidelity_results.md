### Targeted-Categorized

| Target    | Model    | Dataset   | Method   |   fidelity_at_1 |   fidelity_at_5 |   fidelity_at_10 |   fidelity_at_20 |   #users |
|:----------|:---------|:----------|:---------|----------------:|----------------:|-----------------:|-----------------:|---------:|
| Action    | BERT4Rec | ml-100k   | GENE     |           1     |           0.758 |            0.632 |            0.523 |      943 |
| Action    | BERT4Rec | ml-100k   | PACE     |           0.476 |           0.467 |            0.416 |            0.372 |      943 |
| Action    | BERT4Rec | ml-1m     | GENE     |           1     |           0.82  |            0.665 |            0.565 |      200 |
| Action    | BERT4Rec | ml-1m     | PACE     |           0.45  |           0.43  |            0.385 |            0.38  |      200 |
| Action    | GRU4Rec  | ml-100k   | GENE     |           1     |           0.797 |            0.66  |            0.55  |      664 |
| Action    | GRU4Rec  | ml-100k   | PACE     |           0.38  |           0.411 |            0.404 |            0.369 |      664 |
| Action    | SASRec   | ml-100k   | GENE     |           1     |           0.775 |            0.648 |            0.518 |      650 |
| Action    | SASRec   | ml-100k   | PACE     |           0.423 |           0.457 |            0.397 |            0.362 |      650 |
| Adventure | BERT4Rec | ml-100k   | GENE     |           0.997 |           0.594 |            0.353 |            0.209 |      943 |
| Adventure | BERT4Rec | ml-100k   | PACE     |           0.294 |           0.233 |            0.159 |            0.097 |      943 |
| Adventure | BERT4Rec | ml-1m     | GENE     |           1     |           0.66  |            0.44  |            0.305 |      200 |
| Adventure | BERT4Rec | ml-1m     | PACE     |           0.34  |           0.34  |            0.23  |            0.17  |      200 |
| Adventure | GRU4Rec  | ml-100k   | GENE     |           1     |           0.625 |            0.368 |            0.175 |      400 |
| Adventure | GRU4Rec  | ml-100k   | PACE     |           0.248 |           0.24  |            0.155 |            0.09  |      400 |
| Adventure | SASRec   | ml-100k   | GENE     |           1     |           0.632 |            0.42  |            0.242 |      400 |
| Adventure | SASRec   | ml-100k   | PACE     |           0.27  |           0.232 |            0.16  |            0.095 |      400 |
| Animation | BERT4Rec | ml-100k   | GENE     |           0.943 |           0.432 |            0.245 |            0.158 |      943 |
| Animation | BERT4Rec | ml-100k   | PACE     |           0.146 |           0.088 |            0.063 |            0.037 |      943 |
| Animation | BERT4Rec | ml-1m     | GENE     |           0.88  |           0.455 |            0.335 |            0.285 |      200 |
| Animation | BERT4Rec | ml-1m     | PACE     |           0.26  |           0.18  |            0.15  |            0.12  |      200 |
| Animation | GRU4Rec  | ml-100k   | GENE     |           0.975 |           0.252 |            0.08  |            0.025 |      400 |
| Animation | GRU4Rec  | ml-100k   | PACE     |           0.098 |           0.035 |            0.01  |            0.002 |      400 |
| Animation | SASRec   | ml-100k   | GENE     |           0.995 |           0.342 |            0.175 |            0.105 |      400 |
| Animation | SASRec   | ml-100k   | PACE     |           0.16  |           0.072 |            0.048 |            0.03  |      400 |
| Drama     | BERT4Rec | ml-100k   | GENE     |           1     |           0.662 |            0.532 |            0.433 |      476 |
| Drama     | BERT4Rec | ml-100k   | PACE     |           0.62  |           0.481 |            0.414 |            0.342 |      476 |
| Drama     | BERT4Rec | ml-1m     | GENE     |           1     |           0.64  |            0.49  |            0.42  |      200 |
| Drama     | BERT4Rec | ml-1m     | PACE     |           0.54  |           0.41  |            0.365 |            0.3   |      200 |
| Drama     | GRU4Rec  | ml-100k   | GENE     |           1     |           0.59  |            0.49  |            0.405 |      400 |
| Drama     | GRU4Rec  | ml-100k   | PACE     |           0.505 |           0.425 |            0.378 |            0.312 |      400 |
| Drama     | SASRec   | ml-100k   | GENE     |           1     |           0.715 |            0.63  |            0.552 |      400 |
| Drama     | SASRec   | ml-100k   | PACE     |           0.595 |           0.512 |            0.462 |            0.435 |      400 |
| Fantasy   | BERT4Rec | ml-100k   | GENE     |           0.893 |           0.256 |            0.018 |            0     |      943 |
| Fantasy   | BERT4Rec | ml-100k   | PACE     |           0.128 |           0.037 |            0     |            0     |      943 |
| Fantasy   | BERT4Rec | ml-1m     | GENE     |           0.925 |           0.44  |            0.2   |            0.145 |      200 |
| Fantasy   | BERT4Rec | ml-1m     | PACE     |           0.185 |           0.1   |            0.065 |            0.045 |      200 |
| Fantasy   | GRU4Rec  | ml-100k   | GENE     |           0.915 |           0.338 |            0.012 |            0     |      400 |
| Fantasy   | GRU4Rec  | ml-100k   | PACE     |           0.05  |           0.028 |            0.005 |            0     |      400 |
| Fantasy   | SASRec   | ml-100k   | GENE     |           0.915 |           0.375 |            0.025 |            0     |      400 |
| Fantasy   | SASRec   | ml-100k   | PACE     |           0.17  |           0.06  |            0.002 |            0     |      400 |
| Horror    | BERT4Rec | ml-100k   | GENE     |           0.993 |           0.358 |            0.191 |            0.139 |      943 |
| Horror    | BERT4Rec | ml-100k   | PACE     |           0.176 |           0.081 |            0.052 |            0.036 |      943 |
| Horror    | GRU4Rec  | ml-100k   | GENE     |           0.994 |           0.163 |            0.029 |            0.012 |      943 |
| Horror    | GRU4Rec  | ml-100k   | PACE     |           0.103 |           0.028 |            0.012 |            0.012 |      943 |
| Horror    | SASRec   | ml-100k   | GENE     |           0.995 |           0.232 |            0.094 |            0.054 |      943 |
| Horror    | SASRec   | ml-100k   | PACE     |           0.151 |           0.046 |            0.032 |            0.023 |      943 |

### Targeted-Uncategorized

|   Target | Model    | Dataset   | Method   |   fidelity_at_1 |   fidelity_at_5 |   fidelity_at_10 |   fidelity_at_20 |   #users |
|---------:|:---------|:----------|:---------|----------------:|----------------:|-----------------:|-----------------:|---------:|
|     1305 | BERT4Rec | ml-100k   | GENE     |           0.018 |           0.018 |            0.018 |            0.018 |      943 |
|     1305 | BERT4Rec | ml-100k   | PACE     |           0.001 |           0.002 |            0.004 |            0.005 |      943 |
|     1305 | BERT4Rec | ml-1m     | GENE     |           0.058 |           0.058 |            0.058 |            0.058 |      400 |
|     1305 | BERT4Rec | ml-1m     | PACE     |           0.002 |           0.018 |            0.02  |            0.025 |      400 |
|     1305 | GRU4Rec  | ml-100k   | GENE     |           0.01  |           0.01  |            0.01  |            0.01  |      400 |
|     1305 | GRU4Rec  | ml-100k   | PACE     |           0     |           0     |            0     |            0     |      400 |
|     1305 | SASRec   | ml-100k   | GENE     |           0.038 |           0.038 |            0.038 |            0.038 |      400 |
|     1305 | SASRec   | ml-100k   | PACE     |           0.015 |           0.018 |            0.018 |            0.018 |      400 |
|      411 | BERT4Rec | ml-100k   | GENE     |           0.071 |           0.071 |            0.071 |            0.071 |      943 |
|      411 | BERT4Rec | ml-100k   | PACE     |           0.013 |           0.024 |            0.032 |            0.036 |      943 |
|      411 | BERT4Rec | ml-1m     | GENE     |           0.005 |           0.005 |            0.005 |            0.005 |      400 |
|      411 | BERT4Rec | ml-1m     | PACE     |           0.002 |           0.002 |            0.002 |            0.002 |      400 |
|      411 | GRU4Rec  | ml-100k   | GENE     |           0.19  |           0.19  |            0.19  |            0.19  |      506 |
|      411 | GRU4Rec  | ml-100k   | PACE     |           0.03  |           0.04  |            0.045 |            0.053 |      506 |
|      411 | SASRec   | ml-100k   | GENE     |           0.365 |           0.365 |            0.365 |            0.365 |      400 |
|      411 | SASRec   | ml-100k   | PACE     |           0.205 |           0.24  |            0.25  |            0.26  |      400 |
|       50 | BERT4Rec | ml-100k   | GENE     |           0.589 |           0.589 |            0.589 |            0.589 |      943 |
|       50 | BERT4Rec | ml-100k   | PACE     |           0.111 |           0.189 |            0.235 |            0.291 |      943 |
|       50 | BERT4Rec | ml-1m     | GENE     |           0     |           0     |            0     |            0     |      400 |
|       50 | BERT4Rec | ml-1m     | PACE     |           0     |           0     |            0     |            0     |      400 |
|       50 | GRU4Rec  | ml-100k   | GENE     |           0.503 |           0.503 |            0.503 |            0.503 |      943 |
|       50 | GRU4Rec  | ml-100k   | PACE     |           0.043 |           0.11  |            0.156 |            0.206 |      943 |
|       50 | SASRec   | ml-100k   | GENE     |           0.492 |           0.492 |            0.492 |            0.492 |      400 |
|       50 | SASRec   | ml-100k   | PACE     |           0.098 |           0.152 |            0.178 |            0.24  |      400 |
|      630 | BERT4Rec | ml-100k   | GENE     |           0.284 |           0.284 |            0.284 |            0.284 |      943 |
|      630 | BERT4Rec | ml-100k   | PACE     |           0.034 |           0.069 |            0.091 |            0.103 |      943 |
|      630 | BERT4Rec | ml-1m     | GENE     |           0.535 |           0.535 |            0.535 |            0.535 |      400 |
|      630 | BERT4Rec | ml-1m     | PACE     |           0.21  |           0.272 |            0.292 |            0.322 |      400 |
|      630 | GRU4Rec  | ml-100k   | GENE     |           0.148 |           0.148 |            0.148 |            0.148 |      400 |
|      630 | GRU4Rec  | ml-100k   | PACE     |           0.01  |           0.018 |            0.032 |            0.048 |      400 |
|      630 | SASRec   | ml-100k   | GENE     |           0.432 |           0.432 |            0.432 |            0.432 |      400 |
|      630 | SASRec   | ml-100k   | PACE     |           0.128 |           0.175 |            0.188 |            0.192 |      400 |

### Untargeted-Uncategorized

|   Target | Model    | Dataset   | Method   |   fidelity_at_1 |   fidelity_at_5 |   fidelity_at_10 |   fidelity_at_20 |   #users |
|---------:|:---------|:----------|:---------|----------------:|----------------:|-----------------:|-----------------:|---------:|
|      nan | BERT4Rec | ml-100k   | GENE     |           1     |           0.803 |            0.826 |            0.875 |      943 |
|      nan | BERT4Rec | ml-100k   | PACE     |           0.741 |           0.797 |            0.837 |            0.862 |      943 |
|      nan | BERT4Rec | ml-1m     | GENE     |           1     |           0.755 |            0.79  |            0.855 |      200 |
|      nan | BERT4Rec | ml-1m     | PACE     |           0.785 |           0.855 |            0.885 |            0.92  |      200 |
|      nan | GRU4Rec  | ml-100k   | GENE     |           1     |           0.865 |            0.895 |            0.96  |      400 |
|      nan | GRU4Rec  | ml-100k   | PACE     |           0.545 |           0.585 |            0.612 |            0.62  |      400 |
|      nan | SASRec   | ml-100k   | GENE     |           1     |           0.745 |            0.758 |            0.81  |      943 |
|      nan | SASRec   | ml-100k   | PACE     |           0.69  |           0.704 |            0.707 |            0.707 |      943 |

### Untargeted-Categorized

|   Target | Model    | Dataset   | Method   |   fidelity_at_1 |   fidelity_at_5 |   fidelity_at_10 |   fidelity_at_20 |   #users |
|---------:|:---------|:----------|:---------|----------------:|----------------:|-----------------:|-----------------:|---------:|
|      nan | BERT4Rec | ml-100k   | GENE     |           1     |           0.785 |            0.756 |            0.739 |      943 |
|      nan | BERT4Rec | ml-100k   | PACE     |           0.683 |           0.689 |            0.703 |            0.723 |      943 |
|      nan | BERT4Rec | ml-1m     | GENE     |           1     |           0.695 |            0.61  |            0.58  |      200 |
|      nan | BERT4Rec | ml-1m     | PACE     |           0.79  |           0.895 |            0.95  |            0.975 |      200 |
|      nan | GRU4Rec  | ml-100k   | GENE     |           1     |           0.7   |            0.668 |            0.62  |      400 |
|      nan | GRU4Rec  | ml-100k   | PACE     |           0.682 |           0.725 |            0.748 |            0.76  |      400 |
|      nan | SASRec   | ml-100k   | GENE     |           1     |           0.768 |            0.666 |            0.628 |      943 |
|      nan | SASRec   | ml-100k   | PACE     |           0.892 |           0.907 |            0.911 |            0.913 |      943 |


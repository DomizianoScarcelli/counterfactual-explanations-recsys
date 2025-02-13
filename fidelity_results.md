### Targeted-Categorized

| Target    | Model    | Dataset   | Method   |   fidelity@1 |   fidelity@5 |   fidelity@10 |   fidelity@20 |   #users |
|:----------|:---------|:----------|:---------|-------------:|-------------:|--------------:|--------------:|---------:|
| Action    | BERT4Rec | ml-100k   | GENE     |       100    |        76.31 |         64.33 |         53.43 |      743 |
| Action    | BERT4Rec | ml-100k   | PACE     |        46.57 |        46.84 |         41.99 |         38.22 |      743 |
| Action    | BERT4Rec | ml-1m     | GENE     |       100    |        82    |         66.5  |         56.5  |      200 |
| Action    | BERT4Rec | ml-1m     | PACE     |        45    |        43    |         38.5  |         38    |      200 |
| Action    | GRU4Rec  | ml-100k   | GENE     |       100    |        79.5  |         64.5  |         52    |      400 |
| Action    | GRU4Rec  | ml-100k   | PACE     |        38    |        41    |         38.75 |         36.25 |      400 |
| Action    | SASRec   | ml-100k   | GENE     |       100    |        78.75 |         63.5  |         48.75 |      400 |
| Action    | SASRec   | ml-100k   | PACE     |        41    |        44    |         36.75 |         31.75 |      400 |
| Adventure | BERT4Rec | ml-100k   | GENE     |        99.81 |        60.3  |         34.78 |         19.28 |      529 |
| Adventure | BERT4Rec | ml-100k   | PACE     |        28.17 |        21.17 |         14.37 |          8.51 |      529 |
| Adventure | BERT4Rec | ml-1m     | GENE     |       100    |        66    |         44    |         30.5  |      200 |
| Adventure | BERT4Rec | ml-1m     | PACE     |        34    |        34    |         23    |         17    |      200 |
| Adventure | GRU4Rec  | ml-100k   | GENE     |       100    |        62.5  |         36.75 |         17.5  |      400 |
| Adventure | GRU4Rec  | ml-100k   | PACE     |        24.75 |        24    |         15.5  |          9    |      400 |
| Adventure | SASRec   | ml-100k   | GENE     |       100    |        63.25 |         42    |         24.25 |      400 |
| Adventure | SASRec   | ml-100k   | PACE     |        27    |        23.25 |         16    |          9.5  |      400 |
| Animation | BERT4Rec | ml-100k   | GENE     |        96.25 |        43.75 |         25.5  |         17    |      400 |
| Animation | BERT4Rec | ml-100k   | PACE     |        15    |        10.5  |          8    |          4    |      400 |
| Animation | BERT4Rec | ml-1m     | GENE     |        88    |        45.5  |         33.5  |         28.5  |      200 |
| Animation | BERT4Rec | ml-1m     | PACE     |        26    |        18    |         15    |         12    |      200 |
| Animation | GRU4Rec  | ml-100k   | GENE     |        97.5  |        25.25 |          8    |          2.5  |      400 |
| Animation | GRU4Rec  | ml-100k   | PACE     |         9.75 |         3.5  |          1    |          0.25 |      400 |
| Animation | SASRec   | ml-100k   | GENE     |        99.5  |        34.25 |         17.5  |         10.5  |      400 |
| Animation | SASRec   | ml-100k   | PACE     |        16    |         7.25 |          4.75 |          3    |      400 |
| Drama     | BERT4Rec | ml-100k   | GENE     |       100    |        67.25 |         52.5  |         42.75 |      400 |
| Drama     | BERT4Rec | ml-100k   | PACE     |        62.25 |        49    |         42.25 |         34.25 |      400 |
| Drama     | BERT4Rec | ml-1m     | GENE     |       100    |        64    |         49    |         42    |      200 |
| Drama     | BERT4Rec | ml-1m     | PACE     |        54    |        41    |         36.5  |         30    |      200 |
| Drama     | GRU4Rec  | ml-100k   | GENE     |       100    |        59    |         49    |         40.5  |      400 |
| Drama     | GRU4Rec  | ml-100k   | PACE     |        50.5  |        42.5  |         37.75 |         31.25 |      400 |
| Drama     | SASRec   | ml-100k   | GENE     |       100    |        71.5  |         63    |         55.25 |      400 |
| Drama     | SASRec   | ml-100k   | PACE     |        59.5  |        51.25 |         46.25 |         43.5  |      400 |
| Fantasy   | BERT4Rec | ml-100k   | GENE     |        88    |        24    |          1    |          0    |      400 |
| Fantasy   | BERT4Rec | ml-100k   | PACE     |        12.75 |         2    |          0    |          0    |      400 |
| Fantasy   | BERT4Rec | ml-1m     | GENE     |        92.5  |        44    |         20    |         14.5  |      200 |
| Fantasy   | BERT4Rec | ml-1m     | PACE     |        18.5  |        10    |          6.5  |          4.5  |      200 |
| Fantasy   | GRU4Rec  | ml-100k   | GENE     |        91.5  |        33.75 |          1.25 |          0    |      400 |
| Fantasy   | GRU4Rec  | ml-100k   | PACE     |         5    |         2.75 |          0.5  |          0    |      400 |
| Fantasy   | SASRec   | ml-100k   | GENE     |        91.5  |        37.5  |          2.5  |          0    |      400 |
| Fantasy   | SASRec   | ml-100k   | PACE     |        17    |         6    |          0.25 |          0    |      400 |
| Horror    | BERT4Rec | ml-100k   | GENE     |        99.26 |        35.84 |         19.09 |         13.89 |      943 |
| Horror    | BERT4Rec | ml-100k   | PACE     |        17.6  |         8.06 |          5.2  |          3.61 |      943 |
| Horror    | GRU4Rec  | ml-100k   | GENE     |        99    |        17.5  |          2    |          1    |      400 |
| Horror    | GRU4Rec  | ml-100k   | PACE     |         9    |         1.5  |          1    |          1.25 |      400 |
| Horror    | SASRec   | ml-100k   | GENE     |        99.34 |        21.89 |          9.29 |          4.98 |      603 |
| Horror    | SASRec   | ml-100k   | PACE     |        13.27 |         4.15 |          3.15 |          1.99 |      603 |

### Targeted-Uncategorized

|   Target | Model    | Dataset   | Method   |   fidelity@1 |   fidelity@5 |   fidelity@10 |   fidelity@20 |   #users |
|---------:|:---------|:----------|:---------|-------------:|-------------:|--------------:|--------------:|---------:|
|     1305 | BERT4Rec | ml-100k   | GENE     |         1    |         1    |          1    |          1    |      400 |
|     1305 | BERT4Rec | ml-100k   | PACE     |         0    |         0    |          0    |          0    |      400 |
|     1305 | BERT4Rec | ml-1m     | GENE     |         5.75 |         5.75 |          5.75 |          5.75 |      400 |
|     1305 | BERT4Rec | ml-1m     | PACE     |         0.25 |         1.75 |          2    |          2.5  |      400 |
|     1305 | GRU4Rec  | ml-100k   | GENE     |         1    |         1    |          1    |          1    |      400 |
|     1305 | GRU4Rec  | ml-100k   | PACE     |         0    |         0    |          0    |          0    |      400 |
|      411 | BERT4Rec | ml-100k   | GENE     |         8.25 |         8.25 |          8.25 |          8.25 |      400 |
|      411 | BERT4Rec | ml-100k   | PACE     |         1.25 |         3.25 |          4.25 |          5    |      400 |
|      411 | BERT4Rec | ml-1m     | GENE     |         0.5  |         0.5  |          0.5  |          0.5  |      400 |
|      411 | BERT4Rec | ml-1m     | PACE     |         0.25 |         0.25 |          0.25 |          0.25 |      400 |
|      411 | GRU4Rec  | ml-100k   | GENE     |        20    |        20    |         20    |         20    |      400 |
|      411 | GRU4Rec  | ml-100k   | PACE     |         3.75 |         5    |          5.5  |          6.25 |      400 |
|       50 | BERT4Rec | ml-100k   | GENE     |        30.77 |        30.77 |         30.77 |         30.77 |       52 |
|       50 | BERT4Rec | ml-100k   | PACE     |         0    |         3.85 |          7.69 |         13.46 |       52 |
|       50 | BERT4Rec | ml-1m     | GENE     |         0    |         0    |          0    |          0    |      400 |
|       50 | BERT4Rec | ml-1m     | PACE     |         0    |         0    |          0    |          0    |      400 |
|       50 | GRU4Rec  | ml-100k   | GENE     |        43.25 |        43.25 |         43.25 |         43.25 |      400 |
|       50 | GRU4Rec  | ml-100k   | PACE     |         1.75 |         7.25 |         10    |         14.75 |      400 |
|       50 | SASRec   | ml-100k   | GENE     |        49.25 |        49.25 |         49.25 |         49.25 |      400 |
|       50 | SASRec   | ml-100k   | PACE     |         9.75 |        15.25 |         17.75 |         24    |      400 |
|      630 | BERT4Rec | ml-100k   | GENE     |        29.5  |        29.5  |         29.5  |         29.5  |      400 |
|      630 | BERT4Rec | ml-100k   | PACE     |         4.5  |         9.5  |         10.75 |         12.25 |      400 |
|      630 | BERT4Rec | ml-1m     | GENE     |        53.5  |        53.5  |         53.5  |         53.5  |      400 |
|      630 | BERT4Rec | ml-1m     | PACE     |        21    |        27.25 |         29.25 |         32.25 |      400 |
|      630 | GRU4Rec  | ml-100k   | GENE     |        14.75 |        14.75 |         14.75 |         14.75 |      400 |
|      630 | GRU4Rec  | ml-100k   | PACE     |         1    |         1.75 |          3.25 |          4.75 |      400 |

### Untargeted-Uncategorized

|   Target | Model    | Dataset   | Method   |   fidelity@1 |   fidelity@5 |   fidelity@10 |   fidelity@20 |   #users |
|---------:|:---------|:----------|:---------|-------------:|-------------:|--------------:|--------------:|---------:|
|      nan | BERT4Rec | ml-100k   | GENE     |       100    |        77.75 |         79.25 |         86.25 |      400 |
|      nan | BERT4Rec | ml-100k   | PACE     |        73.5  |        79    |         81.75 |         84.25 |      400 |
|      nan | BERT4Rec | ml-1m     | GENE     |       100    |        75.5  |         79    |         85.5  |      200 |
|      nan | BERT4Rec | ml-1m     | PACE     |        78.5  |        85.5  |         88.5  |         92    |      200 |
|      nan | GRU4Rec  | ml-100k   | GENE     |       100    |        86.5  |         89.5  |         96    |      400 |
|      nan | GRU4Rec  | ml-100k   | PACE     |        54.5  |        58.5  |         61.25 |         62    |      400 |
|      nan | SASRec   | ml-100k   | GENE     |       100    |        73.31 |         76.69 |         80.96 |      562 |
|      nan | SASRec   | ml-100k   | PACE     |        68.68 |        70.28 |         70.64 |         70.64 |      562 |

### Untargeted-Categorized

|   Target | Model    | Dataset   | Method   |   fidelity@1 |   fidelity@5 |   fidelity@10 |   fidelity@20 |   #users |
|---------:|:---------|:----------|:---------|-------------:|-------------:|--------------:|--------------:|---------:|
|      nan | BERT4Rec | ml-100k   | GENE     |       100    |        89.25 |         90.5  |         93.75 |      400 |
|      nan | BERT4Rec | ml-100k   | PACE     |        51.25 |        45.5  |         44    |         44    |      400 |
|      nan | BERT4Rec | ml-1m     | GENE     |       100    |        69.5  |         61    |         58    |      200 |
|      nan | BERT4Rec | ml-1m     | PACE     |        79    |        89.5  |         95    |         97.5  |      200 |
|      nan | GRU4Rec  | ml-100k   | GENE     |       100    |        70    |         66.75 |         62    |      400 |
|      nan | GRU4Rec  | ml-100k   | PACE     |        68.25 |        72.5  |         74.75 |         76    |      400 |
|      nan | SASRec   | ml-100k   | GENE     |       100    |        74.75 |         66.25 |         60.75 |      400 |
|      nan | SASRec   | ml-100k   | PACE     |        88    |        89.75 |         90    |         90    |      400 |


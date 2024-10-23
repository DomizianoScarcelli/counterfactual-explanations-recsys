The format is ["log"|"stats"]_[pop_size]_[num_gens].json.

- `hop_h/`: contains the logs and the stats for the trace alignment with the hop heuristic with bfs. 
    - Note for log_2000_10: use_cache was True, but some datasets had to be generated
    - This is the fastest for now in generating the counterfactuals
- `no_h/`: contains the logs and the stats for the trace alignment with Dijkstra (no heuristic).
    - Note for log_2000_10: use_cache was True, and all datasets were already generated.
- `add_dels_h/`: the heuristic returns 0 if the adds + dels operations are divisible by 2, the number of adds + dels otherwise.
    - Note for log_2000_10: use_cache was True, and all datasets were already generated.

#!bin/bash
eval_path="results/evaluate/alignment/evaluation_all.csv"
fidelity_path="results/stats/alignment/fidelity_all.csv"

python -m cli scripts merge_dfs --primary-key="['i', 'gen_source', 'gen_target_y@1', 'generation_strategy']" \
    --blacklist-keys="['timestamp', 'target_cat']" \
    --dir="results/evaluate/alignment/" \
    --regex="^alignment.*" \
    --save-path=$eval_path

python -m cli stats fidelity --log-path=$eval_path --save-path=$fidelity_path

vd --play "scripts/visidata/filter_fidelity_all.vdj"

python -m scripts.fidelity_all_to_md

nvim fidelity_results.md

python -m scripts.check_fidelity_progress


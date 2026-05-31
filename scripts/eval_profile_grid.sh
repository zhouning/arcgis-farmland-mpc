#!/bin/bash
# Evaluate the 8 (case × profile) cells: 3-seed lam=5 plan, 3-seed lam=0 plan,
# OR baselines (greedy/SA/NSGA2/MILP), random baseline, and ranking metrics.

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate farmland-mpc

eval_cell() {
    local case=$1 profile=$2
    local prep=runs/restoration/$case/prepared_${profile}
    local out_root=runs/restoration/$case/profiles/${profile}
    mkdir -p $out_root

    # Pick the right attribute file for OR baselines
    case $case in
        buchanan_va) attrs=runs/restoration/$case/planning_units_2km_attributes.csv;;
        synthetic)   attrs=runs/restoration/$case/restoration_units_attributes.csv;;
    esac

    echo "[$(date '+%H:%M:%S')] === $case / $profile ==="

    # 3-seed lam=5 plan
    python -m farmland_mpc.tests.eval_5seed_paper \
        --prepared-dir $prep \
        --out-json $out_root/3seed_lam5.json \
        --region "${case}-${profile}-lam5" \
        --ensemble-prefix ensemble_seed --n-seeds 3 --n-episodes-per-seed 1 \
        --continuation greedy --env restoration --lambda-rank 5.0 2>&1 | tail -1

    # 3-seed lam=0 plan
    python -m farmland_mpc.tests.eval_5seed_paper \
        --prepared-dir $prep \
        --out-json $out_root/3seed_lam0.json \
        --region "${case}-${profile}-lam0" \
        --ensemble-prefix ensemble_lam0_seed --n-seeds 3 --n-episodes-per-seed 1 \
        --continuation greedy --env restoration --lambda-rank 0.0 2>&1 | tail -1

    # Random baseline
    python /tmp/random_baseline.py $prep $out_root/random.json 5

    # OR baselines (greedy/SA/NSGA-II/MILP)
    python -m farmland_mpc.tests.or_baselines \
        --prepared-dir $prep \
        --units-attributes $attrs \
        --out-dir $out_root/or_baselines \
        --case ${case%_va} 2>&1 | tail -3

    # Ranking metrics on lam=5 seed0 and lam=0 seed0
    n_blocks=$(python3 -c "import json; print(json.load(open('$prep/scenario_config.json'))['n_units'])")
    python -m farmland_mpc.tests.eval_ranking_metrics \
        --ensemble-dir $prep/ensemble_seed0 \
        --pairwise $prep/tool2/pairwise.npz \
        --n-blocks $n_blocks --label "$case-$profile-lam5" \
        --out-json $out_root/ranking_lam5.json 2>&1 | tail -1
    python -m farmland_mpc.tests.eval_ranking_metrics \
        --ensemble-dir $prep/ensemble_lam0_seed0 \
        --pairwise $prep/tool2/pairwise.npz \
        --n-blocks $n_blocks --label "$case-$profile-lam0" \
        --out-json $out_root/ranking_lam0.json 2>&1 | tail -1

    echo "[$(date '+%H:%M:%S')] done $case / $profile"
}

for case in buchanan_va synthetic; do
    for profile in connectivity_dominant watershed scale_economy delayed; do
        eval_cell $case $profile
    done
done

echo "ALL DONE"
date

#!/bin/bash
# Budget-matched comparison across all 10 (case × profile) cells.
# For each cell run:
#   - SA at iteration counts {50, 100, 200, 300, 500}
#   - MPC with random continuation, 50 episodes (~2,500 env.step calls)
# Both methods are then compared at matched env.step budgets.

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate farmland-mpc

run_cell() {
    local case=$1 profile=$2
    if [ "$profile" = "default" ]; then
        local prep=runs/restoration/$case/prepared
        local out_root=runs/restoration/$case/budget_matched
    else
        local prep=runs/restoration/$case/prepared_${profile}
        local out_root=runs/restoration/$case/profiles/${profile}/budget_matched
    fi
    mkdir -p $out_root

    # Pick attrs file
    case $case in
        buchanan_va) local attrs=runs/restoration/$case/planning_units_2km_attributes.csv;;
        synthetic)   local attrs=runs/restoration/$case/restoration_units_attributes.csv;;
    esac

    echo "[$(date '+%H:%M:%S')] === $case / $profile ==="

    # SA at multiple iteration budgets
    for ITERS in 50 100 200 300 500; do
        python /tmp/sa_budget.py $prep $attrs $out_root/sa_iter${ITERS}.json $ITERS 2>&1 | tail -1
    done

    # MPC random continuation, 50 episodes
    python -m farmland_mpc.tests.eval_mpc_multi_ep \
        --prepared-dir $prep \
        --ensemble-dir $prep/ensemble_seed0 \
        --n-episodes 50 --horizon 5 --top-k 50 --continuation random \
        --out-json $out_root/mpc_random_h5_50ep.json 2>&1 | tail -1

    echo "[$(date '+%H:%M:%S')] done $case / $profile"
}

for case in buchanan_va synthetic; do
    for profile in default connectivity_dominant watershed scale_economy delayed; do
        run_cell $case $profile
    done
done
echo "ALL DONE"
date

#!/bin/bash
# Train 3 lam=5 + 3 lam=0 ensembles per (case, profile) combination,
# 4 profiles × 2 cases = 8 prepared dirs, 6 ensembles each = 48 ensembles total.
# 3 ensembles run in parallel via the existing --torch-threads 4 setup.
# Wall-clock estimate: ~50 min on a 14-core M-series Mac.

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate farmland-mpc

train_one() {
    prep=$1; lam=$2; seed=$3
    if (( $(echo "$lam == 0.0" | bc -l) )); then
        sub="ensemble_lam0_seed${seed}"
    else
        sub="ensemble_seed${seed}"
    fi
    farmland-mpc train \
        --prepared-dir "$prep" \
        --n-members 3 --epochs 30 --patience 8 \
        --lambda-rank $lam --margin 0.1 \
        --seed-base $seed --torch-threads 4 \
        --out-subdir $sub 2>&1 | tail -1
}

run_3parallel() {
    "$@" &
}

for case in buchanan_va synthetic; do
    for profile in connectivity_dominant watershed scale_economy delayed; do
        prep=runs/restoration/$case/prepared_${profile}
        echo "=== $case / $profile ==="
        # batch1: 3 lam=5 ensembles
        for SEED in 0 1 2; do
            train_one $prep 5.0 $SEED &
        done
        wait
        # batch2: 3 lam=0 ensembles
        for SEED in 0 1 2; do
            train_one $prep 0.0 $SEED &
        done
        wait
        date "+  %H:%M:%S done $case / $profile"
    done
done
echo "ALL DONE"
date

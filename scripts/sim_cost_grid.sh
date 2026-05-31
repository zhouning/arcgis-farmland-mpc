#!/bin/bash
# Run sim_cost_sweep on the 7 remaining (case × profile) cells.
# Already done: buchanan/default, buchanan/delayed, synthetic/scale_economy.

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate farmland-mpc

run_cell() {
    local case=$1 profile=$2
    if [ "$profile" = "default" ]; then
        local prep=runs/restoration/$case/prepared
        local out=runs/restoration/$case/sim_cost_sweep.json
    else
        local prep=runs/restoration/$case/prepared_${profile}
        local out=runs/restoration/$case/profiles/${profile}/sim_cost_sweep.json
    fi
    case $case in
        buchanan_va) local attrs=runs/restoration/$case/planning_units_2km_attributes.csv  ;;
        synthetic)   local attrs=runs/restoration/$case/restoration_units_attributes.csv  ;;
    esac

    if [ -f "$out" ]; then
        echo "[$(date '+%H:%M:%S')] skip (exists): $case/$profile"
        return
    fi

    local case_arg=${case%_va}
    echo "[$(date '+%H:%M:%S')] === $case / $profile ==="
    python -m farmland_mpc.tests.simulator_cost_sweep \
        --prepared-dir $prep \
        --ensemble-dir $prep/ensemble_seed0 \
        --units-attributes $attrs \
        --out-json $out \
        --case $case_arg 2>&1 | tail -10
    echo "[$(date '+%H:%M:%S')] done $case/$profile"
}

# Skip already-completed: buchanan_va/default, buchanan_va/delayed, synthetic/scale_economy
for case in buchanan_va synthetic; do
    for profile in default connectivity_dominant watershed scale_economy delayed; do
        run_cell $case $profile
    done
done
echo "ALL DONE"
date

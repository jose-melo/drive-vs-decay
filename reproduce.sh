#!/usr/bin/env bash
# Reproduce one named experiment of the paper.
#   bash reproduce.sh --list
#   bash reproduce.sh <name> [--show | --cell N]
# Image experiments are forwarded to image/scripts/run.sh (see its header for DATA_DIR, OUT_DIR, LAUNCH).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TABULAR="phase-grid-aloi phase-grid-helena phase-grid-jannis tabular-residualpred"

list() {
    echo "Tabular (run from tabular/):"
    echo "  phase-grid-<aloi|helena|jannis>  Figure 3 / Table 7: linear Tabular-JEPA and T-JEPA grids over predictor scale x context ratio"
    echo "  tabular-residualpred             Table 4, tabular columns: T-JEPA vs T-JEPA + ResidualPred, 5 cells x 3 seeds x 2 predictors"
    echo "Image (run from image/):"
    bash image/scripts/run.sh --list | sed -n '/^Experiments:/,$p' | tail -n +2
}

tabular_residualpred() {
    local cells=("jannis 0.075 0.2" "jannis 0.15 0.5" "jannis 0.15 1.0" "helena 0.15 0.5" "helena 0.15 1.0")
    local c ds ctx scale seed variant
    for c in "${cells[@]}"; do
        read -r ds ctx scale <<< "$c"
        for seed in 0 1 2; do
            for variant in standard residualpred; do
                local flag=""; [[ $variant == residualpred ]] && flag="--use_residual_predictor"
                echo "python experiment_2.py --experiment single_point --data_set $ds --context_ratio $ctx" \
                     "--pred_init_scale $scale --seed $seed $flag" \
                     "--output_dir experiments/results/tabular_residualpred/$ds/$variant/seed$seed"
            done
        done
    done
}

[[ $# -eq 0 || "$1" == "--list" || "$1" == "-h" ]] && { list; exit 0; }
NAME="$1"; shift
case "$NAME" in
    phase-grid-*)
        cd tabular && bash reproduce_tabular.sh --dataset "${NAME#phase-grid-}" ;;
    tabular-residualpred)
        if [[ "${1:-}" == "--show" ]]; then tabular_residualpred; exit 0; fi
        cd tabular && bash reproduce_tabular.sh --download-only
        tabular_residualpred | while read -r cmd; do echo "$cmd"; eval "$cmd"; done ;;
    *)
        exec bash image/scripts/run.sh "$NAME" "$@" ;;
esac

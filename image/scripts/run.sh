#!/usr/bin/env bash
# Image experiments of the paper, one named experiment per call.
#
#   bash scripts/run.sh --list                     experiments and their size
#   bash scripts/run.sh in1k-pilot --show          print every command of an experiment
#   bash scripts/run.sh in1k-pilot                 run every cell, one after the other
#   bash scripts/run.sh in1k-pilot --cell 3        run one cell (for SLURM arrays)
#
# Environment:
#   DATA_DIR    where the datasets live (default ./data): cifar-10-batches-py, cifar-100-python,
#               stl10_binary, imagenet100/{train,val}, imagenet/{train,val}
#   OUT_DIR     where runs are written (default ./logs)
#   LAUNCH      prefix for the python call, e.g. "srun" for multi-GPU SLURM jobs (default empty)
#   EPOCHS      override the number of epochs (smoke tests)
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SELF")/.."
DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./logs}"
LAUNCH="${LAUNCH:-}"

# config|dataset|root|predictor|depth|seed|mask scale|flags
arm_flags() {
    case "$1" in
        standard)       echo "standard|" ;;
        residualpred)   echo "residualpred|" ;;
        bias_only)      echo "residualpred|--no_zero_init" ;;
        zero_init_only) echo "residualpred|--beta_init 0" ;;
    esac
}

cifar_cell() {  # config dataset arm seed [extra]
    local pf; pf="$(arm_flags "$3")"
    echo "$1|$2|$DATA_DIR|${pf%%|*}|4|$4|0.8 0.9|${pf#*|}${5:+ $5}"
}

cells() {
    local s v d l
    case "$1" in
        cifar10|stl10)
            local cfg=configs/cifar10_vits4_ep100.yaml
            [[ "$1" == stl10 ]] && cfg=configs/stl10_vits8_ep100.yaml
            for v in standard residualpred bias_only zero_init_only; do for s in 0 1 2; do
                cifar_cell $cfg "$1" $v $s; done; done ;;
        cifar100)
            for v in standard residualpred zero_init_only; do for s in 0 7 42 123 314; do
                cifar_cell configs/cifar100_vits4_ep100.yaml cifar100 $v $s; done; done
            for s in 0 1 2; do cifar_cell configs/cifar100_vits4_ep100.yaml cifar100 bias_only $s; done ;;
        cifar100-300ep)
            for s in 0 1; do cifar_cell configs/cifar100_vits4_ep100.yaml cifar100 standard $s "--epochs 300"; done
            for v in residualpred bias_only zero_init_only; do for s in 0 1 2; do
                cifar_cell configs/cifar100_vits4_ep100.yaml cifar100 $v $s "--epochs 300"; done; done ;;
        in100)
            for v in standard residualpred; do for d in 2 6; do for s in 0 1 2; do
                echo "configs/in100_vits16_ep150.yaml|imagenet100|$DATA_DIR/imagenet100|$v|$d|$s|0.85 1.0|"; done; done; done ;;
        in100-vitb)
            for v in standard residualpred; do for s in 0 1 2; do
                echo "configs/in100_vitb16_ep100.yaml|imagenet100|$DATA_DIR/imagenet100|$v|6|$s|0.85 1.0|"; done; done ;;
        in100-controls)
            for v in frozen_rp skiponly; do for s in 0 1 2; do
                echo "configs/in100_vits16_ep150.yaml|imagenet100|$DATA_DIR/imagenet100|$v|6|$s|0.85 1.0|"; done; done ;;
        in1k-pilot)
            for v in standard residualpred; do for s in 0 1 2 3 4; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|$v|6|$s|0.85 1.0|"; done; done ;;
        in1k-224px)
            for v in standard residualpred; do for s in 0 1 2 3 4; do
                echo "configs/in1k_vits16_224px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|$v|6|$s|0.85 1.0|"; done; done ;;
        in1k-90ep)
            for v in standard residualpred; do for s in 0 1 2; do
                echo "configs/in1k_vits16_128px_90ep.yaml|imagenet1k|$DATA_DIR/imagenet|$v|6|$s|0.85 1.0|"; done; done ;;
        in1k-sigreg)
            for s in 0 1 2 3 4 5 6; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|standard|6|$s|0.85 1.0|--sigreg_lambda 0.05"; done
            for l in 0.02 0.15; do for s in 0 1; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|standard|6|$s|0.85 1.0|--sigreg_lambda $l"; done; done
            for s in 0 1 2 3 4; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|residualpred|6|$s|0.85 1.0|--sigreg_lambda 0.05"; done ;;
        in1k-vicreg)
            for s in 0 1; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|standard|6|$s|0.85 1.0|--vicreg_var_weight 1.0 --vicreg_cov_weight 0.04"; done ;;
        in1k-mimetic)
            for s in 0 1 2; do
                echo "configs/in1k_vits16_128px_30ep.yaml|imagenet1k|$DATA_DIR/imagenet|mimetic|6|$s|0.85 1.0|--mimetic_alpha 0.5"; done ;;
        *) return 1 ;;
    esac
}

EXPERIMENTS="cifar10 cifar100 stl10 cifar100-300ep in100 in100-vitb in100-controls in1k-pilot in1k-224px in1k-90ep in1k-sigreg in1k-vicreg in1k-mimetic"

usage() {
    sed -n '2,14p' "$SELF" | sed 's/^# \{0,1\}//'
    echo "Experiments:"
    for e in $EXPERIMENTS; do printf '  %-16s %3d runs\n' "$e" "$(cells "$e" | wc -l)"; done
}

command_for() {
    local config dataset root predictor depth seed mask extra tag
    IFS='|' read -r config dataset root predictor depth seed mask extra <<< "$1"
    tag="${2}_${predictor}_d${depth}_seed${seed}"
    local flags; flags="$(echo "$extra" | sed 's/--epochs [0-9]*//' | tr -d '-' | xargs | tr ' ' '_')"
    [[ -n "$flags" ]] && tag="${tag}_${flags}"
    echo "$LAUNCH python -m src.train --config $config --dataset $dataset --root_path $root" \
         "--predictor_type $predictor --pred_depth $depth --seed $seed" \
         "--enc_mask_scale_low ${mask% *} --enc_mask_scale_high ${mask#* }" \
         "--tag $tag --folder $OUT_DIR/$2/$tag/ --checkpoint_freq 0" \
         "$extra${EPOCHS:+ --epochs $EPOCHS}"
}

[[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" || "$1" == "--list" ]] && { usage; exit 0; }
EXP="$1"; shift
cells "$EXP" > /dev/null || { echo "unknown experiment: $EXP" >&2; usage; exit 2; }
CELLS=()
while IFS= read -r line; do CELLS+=("$line"); done < <(cells "$EXP")

MODE=run; CELL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --show) MODE=show; shift ;;
        --cell) CELL="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

IDX=("${!CELLS[@]}")
[[ -n "$CELL" ]] && IDX=("$CELL")
for i in "${IDX[@]}"; do
    [[ -n "${CELLS[$i]:-}" ]] || { echo "cell $i out of range (0..$(( ${#CELLS[@]} - 1 )))" >&2; exit 2; }
    cmd="$(command_for "${CELLS[$i]}" "$EXP")"
    if [[ "$MODE" == show ]]; then echo "[$i] $cmd"; else echo "[$i] $cmd"; eval "$cmd"; fi
done

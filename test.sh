#!/usr/bin/env bash
# Smoke test: a few minutes on one GPU. Downloads helena (OpenML) and CIFAR-10 if absent,
# runs one short tabular run per pipeline and one short image run per predictor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
OUT="${OUT:-$(pwd)/smoke}"
mkdir -p "$OUT"

echo "== tabular: data"
(cd tabular && bash reproduce_tabular.sh --download-only)

echo "== tabular: linear Tabular-JEPA, one grid point"
(cd tabular && python experiment_1.py --experiment single_point --data_set helena --data_path data \
    --num_epochs 2 --probe_cadence 2 --probe_epochs 2 --output_dir "$OUT/linear")

for variant in "" "--use_residual_predictor"; do
    echo "== tabular: T-JEPA ${variant:-standard}"
    (cd tabular && python experiment_2.py --experiment single_point --data_set helena \
        --context_ratio 0.15 --pred_init_scale 1.0 --seed 0 --num_epochs 2 --probe_cadence 2 \
        $variant --output_dir "$OUT/tjepa/${variant:-standard}")
done

echo "== image: data"
(cd image && python scripts/download_datasets.py --root data --datasets cifar10)

for predictor in standard residualpred frozen_rp skiponly mimetic; do
    echo "== image: CIFAR-10, $predictor, 1 epoch"
    (cd image && python -m src.train --config configs/cifar10_vits4_ep100.yaml --dataset cifar10 --root_path data \
        --predictor_type "$predictor" --pred_depth 4 --enc_mask_scale_low 0.8 --enc_mask_scale_high 0.9 \
        --seed 0 --epochs 1 --tag "smoke_$predictor" --folder "$OUT/image/$predictor/" --checkpoint_freq 0)
done

echo "== image: SIGReg and VICReg terms, 1 epoch"
(cd image && python -m src.train --config configs/cifar10_vits4_ep100.yaml --dataset cifar10 --root_path data \
    --predictor_type residualpred --pred_depth 4 --seed 0 --epochs 1 --sigreg_lambda 0.05 \
    --vicreg_var_weight 1.0 --vicreg_cov_weight 0.04 --tag smoke_aux --folder "$OUT/image/aux/" --checkpoint_freq 0)

echo "== results summary"
python results/summarize_image_results.py > /dev/null && echo "summarize_image_results.py OK"
echo "smoke test passed; outputs in $OUT"

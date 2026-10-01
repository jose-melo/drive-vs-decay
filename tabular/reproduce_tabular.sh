#!/usr/bin/env bash
set -euo pipefail

DATASETS=("aloi" "helena" "jannis")
OUTPUT_DIR="./experiments/results"
mkdir -p "$OUTPUT_DIR" data

LINEAR_ONLY=false
TJEPA_ONLY=false
DOWNLOAD_ONLY=false
SUBSET_DATASET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --linear-only)   LINEAR_ONLY=true; shift ;;
        --tjepa-only)    TJEPA_ONLY=true; shift ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        --dataset)       SUBSET_DATASET="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: reproduce_tabular.sh [--linear-only] [--tjepa-only] [--download-only] [--dataset NAME]
EOF
            exit 0
            ;;
        *) echo "Unknown option $1"; exit 2 ;;
    esac
done

if [[ -n "$SUBSET_DATASET" ]]; then
    DATASETS=("$SUBSET_DATASET")
fi

download_data() {
    if [[ ! -f data/aloi.csv ]]; then
        python3 - <<'PY'
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
data = fetch_openml(data_id=1592, as_frame=False, parser='auto')
X = data.data.toarray() if hasattr(data.data, 'toarray') else np.asarray(data.data)
df = pd.DataFrame(X, columns=data.feature_names)
df['class'] = data.target
df.to_csv('data/aloi.csv', index=False)
PY
    fi

    # name, OpenML id
    set -- helena 41169 jannis 41168
    while [[ $# -ge 2 ]]; do
        name="$1"; oid="$2"; shift 2
        if [[ ! -f "data/${name}.arff" ]]; then
            python3 - "$name" "$oid" <<'PY'
import sys
import pandas as pd
from sklearn.datasets import fetch_openml

name, oid = sys.argv[1], int(sys.argv[2])
data = fetch_openml(data_id=oid, as_frame=True, parser='auto')
df = data.frame.copy()

# integer class labels, as declared in the ARFF header
if 'class' in df.columns:
    df['class'] = pd.to_numeric(df['class'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['class']).reset_index(drop=True)
    df['class'] = df['class'].astype(int)

with open(f'data/{name}.arff', 'w') as f:
    f.write(f'@RELATION {name}\n\n')
    for col in df.columns:
        if col == 'class':
            classes = ','.join(str(int(v)) for v in sorted(df[col].unique()))
            f.write(f'@ATTRIBUTE class {{{classes}}}\n')
        else:
            f.write(f'@ATTRIBUTE {col} NUMERIC\n')
    f.write('\n@DATA\n')
    for _, row in df.iterrows():
        parts = []
        for col, v in zip(df.columns, row.values):
            if col == 'class':
                parts.append(str(int(v)))
            else:
                parts.append(repr(float(v)))
        f.write(','.join(parts) + '\n')
print(f'Wrote data/{name}.arff')
PY
        fi
    done
}

run_linear_tjepa() {
    for dataset in "${DATASETS[@]}"; do
        python experiment_1.py \
            --experiment critical_2d \
            --data_set "$dataset" \
            --data_path data \
            --output_dir "$OUTPUT_DIR/linear_tjepa/$dataset" \
            --num_epochs 30 \
            --probe_cadence 10
    done
}

run_tjepa() {
    for dataset in "${DATASETS[@]}"; do
        python experiment_2.py \
            --experiment critical_2d \
            --data_set "$dataset" \
            --output_dir "$OUTPUT_DIR/tjepa/$dataset" \
            --probe_cadence 10
    done
}

download_data
[[ "$DOWNLOAD_ONLY" == true ]] && exit 0

if [[ "$LINEAR_ONLY" == true ]]; then
    run_linear_tjepa
elif [[ "$TJEPA_ONLY" == true ]]; then
    run_tjepa
else
    run_linear_tjepa
    run_tjepa
fi

#!/usr/bin/env bash
# Execute all tutorial notebooks in understanding_its2s/ with outputs stored in place.
# Run from the repo root.
#
# Prerequisites:
#   pip install -e ".[neural]"
#   pip install jupyter nbconvert
set -euo pipefail

TUTORIALS="understanding_its2s"

cd "$TUTORIALS"

for nb in \
    step1_data_splitting.ipynb \
    step2_cross_validation.ipynb \
    step3_hyperparameter_tuning.ipynb \
    step4a_model_prophet_xgb.ipynb \
    step4b_model_prophet_then_xgb.ipynb \
    step4c_model_neuralprophet.ipynb \
    step4d_model_arima.ipynb \
    step5_bootstrap_mbb.ipynb \
    step6_full_workflow.ipynb; do
    echo "Executing $nb ..."
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 "$nb"
done

echo ""
echo "All notebooks executed. Stage changes with:"
echo "  git add understanding_its2s/*.ipynb understanding_its2s/figures/"

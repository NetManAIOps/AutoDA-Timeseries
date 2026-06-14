# AutoDA-Timeseries

This repository contains the cleaned AutoDA-Timeseries implementation for time-series data augmentation experiments.

Project website: https://netmanaiops.github.io/AutoDA-Timeseries/

## Included

- `autoaugment/AutoDA_Timeseries.py`: AutoDA-Timeseries augmentation policy.
- `autoaugment/augments/basic_transforms.py`: time-series augmentation operators used by AutoDA.
- `exp/`: training and evaluation loops for classification, regression, forecasting, and anomaly detection.
- `dataloader/`: dataset adapters and feature extraction.
- `downstream/`: downstream models.
- `examples/AutoDA-Timeseries.classification.json`: example AutoDA configuration.

Historical augmentation baselines and backup files are intentionally not included.

## Example

```bash
source /workspace/douzj/AutoTSA-master/douzj_AutoTSA/bin/activate
python -u run.py \
  --task classification \
  --tsa AutoDA-Timeseries \
  --downstream ROCKET \
  --feature_extractor Catch22 \
  --tsa_config_path examples/AutoDA-Timeseries.classification.json \
  --dataset_root /workspace/douzj/AutoTSA-master/dataset \
  --dataset ArticularyWordRecognition \
  --dataset_type Default \
  --use_gpu 1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --train_epochs 35 \
  --patience 17
```

The classification helper script is available at:

```bash
bash scripts/classification/AutoDA_Timeseries.classification_distri.sh
```

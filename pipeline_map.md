# IEH Pipeline Map

Generated: 2026-06-01

## Official Pipeline

```text
Data Collection
  scripts/ieh_universal_harvester.py
    -> yashas_data/mouse_raw.parquet
    -> yashas_data/keyboard_raw.parquet
    -> yashas_data/harvester_v2.log

Feature Extraction
  universal_feature_extractor.py
    <- yashas_data/*.parquet
    <- imposters/*.parquet
    -> master_feature_dataset.parquet

Training
  scripts/trainer_engine.py
    <- master_feature_dataset.parquet
    <- scripts/model_architecture.py
    -> checkpoints/scaler.pkl
    -> checkpoints/best_model.pth

Calibration
  scripts/threshold_calibrator.py
    <- master_feature_dataset.parquet
    <- checkpoints/scaler.pkl
    <- checkpoints/best_model.pth
    <- scripts/model_architecture.py

Inference
  future
```

## Verification

| Link | Status |
|---|---|
| Harvester output -> extractor input | OK: harvester writes `yashas_data`, extractor reads `yashas_data`. |
| Extractor output -> trainer input | OK: extractor writes `master_feature_dataset.parquet`, trainer reads same file. |
| Trainer output -> calibrator input | OK: trainer writes checkpoints, calibrator reads `checkpoints/best_model.pth` and `checkpoints/scaler.pkl`. |

## Known Non-Pipeline Historical Data

`archive/legacy_data/personal_snapshot.parquet` contains large owner-like raw telemetry but uses a different schema:

```text
timestamp, event_type, x, y
```

It should remain archived until an adapter is created and validated. It should not be blindly copied into `yashas_data/`.


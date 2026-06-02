# AC IEH Prototype

Adaptive Continuous Implicit Human Authentication prototype.

This repo contains the runnable Streamlit workspace, trained checkpoint,
scaler, trust/session engines, and small experiment fixtures needed to demo
owner, impostor, hijack, persistent attack, and recovery flows.

## Quick Start

```powershell
git clone https://github.com/YASHAS-MN/AC_IEH.git
cd AC_IEH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_workspace.py
```

## Prototype Entry Point

```powershell
streamlit run streamlit_workspace.py
```

The workspace uses:

- `checkpoints/best_model.pth`
- `checkpoints/scaler.pkl`
- `experiment/owner_test.parquet`
- `experiment/impostor_test.parquet`
- `scripts/authentication_pipeline.py`
- `scripts/inference_engine.py`
- `scripts/session_engine.py`
- `scripts/trust_engine_v2.py`

## Demo Scenarios

Use the sidebar `Demo Scenario` panel:

- `Owner Session`
- `Session Hijack`
- `Persistent Attack`
- `Recovery Flow`

The app writes local behavioral decisions to `session_audit.csv`. That file is
runtime output and is intentionally ignored by Git.

## Repository Hygiene

The repo intentionally excludes:

- `.venv/`
- Python caches
- raw private owner telemetry in `yashas_data/`
- raw impostor collections in `imposters/`
- archives/backups/evaluation output
- local runtime logs and audit CSVs

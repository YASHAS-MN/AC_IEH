from pathlib import Path

import joblib
import pandas as pd
import torch

try:
    from scripts.model_architecture import create_model
except ModuleNotFoundError:
    from model_architecture import create_model


ROOT = Path(__file__).resolve().parent.parent

MODEL = ROOT / "checkpoints" / "best_model.pth"
SCALER = ROOT / "checkpoints" / "scaler.pkl"

THRESHOLD = 0.200663

FEATURES = [
    "velocity_mean",
    "velocity_std",
    "velocity_max",
    "acceleration_mean",
    "acceleration_std",
    "jerk_mean",
    "jerk_std",
    "curvature_mean",
    "curvature_std",
    "trajectory_entropy",
    "dwell_mean",
    "dwell_std",
    "flight_mean",
    "flight_std",
]


class InferenceEngine:
    def __init__(self):
        self.model = create_model(len(FEATURES))
        self.model.load_state_dict(
            torch.load(
                MODEL,
                map_location="cpu",
            )
        )
        self.model.eval()

        self.scaler = joblib.load(SCALER)

    def predict(self, df: pd.DataFrame) -> dict:
        x = df[FEATURES].copy()
        x = self.scaler.transform(x)
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            reconstruction = self.model(x)
            error = ((reconstruction - x) ** 2).mean(dim=1)

        score = float(error.mean())

        return {
            "verified": score < THRESHOLD,
            "score": round(score, 6),
        }


if __name__ == "__main__":
    sample = pd.read_parquet(
        ROOT / "experiment" / "owner_test.parquet"
    ).head(50)

    engine = InferenceEngine()
    print(engine.predict(sample))

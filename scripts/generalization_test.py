import json
import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from model_architecture import BehavioralAutoencoder


DEVICE="cpu"

MODEL="checkpoints/best_model.pth"
SCALER="checkpoints/scaler.pkl"

TRAIN="experiment/owner_train.parquet"
OWNER="experiment/owner_test.parquet"
IMP="experiment/impostor_test.parquet"

FEATURES=[
'velocity_mean',
'velocity_std',
'velocity_max',
'acceleration_mean',
'acceleration_std',
'jerk_mean',
'jerk_std',
'curvature_mean',
'curvature_std',
'trajectory_entropy',
'dwell_mean',
'dwell_std',
'flight_mean',
'flight_std'
]


def score(model,x):

    with torch.no_grad():

        x=torch.tensor(
            x,
            dtype=torch.float32
        )

        recon=model(x)

        err=(
            (
                recon-x
            )**2
        ).mean(
            dim=1
        )

        return (
            err
            .cpu()
            .numpy()
        )


def main():

    scaler=joblib.load(
        SCALER
    )

    model=BehavioralAutoencoder(
        input_dim=len(FEATURES)
    )

    model.load_state_dict(
        torch.load(
            MODEL,
            map_location=DEVICE
        )
    )

    model.eval()

    owner=pd.read_parquet(
        OWNER
    )

    imp=pd.read_parquet(
        IMP
    )

    xo=scaler.transform(
        owner[
            FEATURES
        ]
    )

    xi=scaler.transform(
        imp[
            FEATURES
        ]
    )

    so=score(
        model,
        xo
    )

    si=score(
        model,
        xi
    )

    threshold=np.percentile(
        so,
        99
    )

    owner_accept=(
        so<threshold
    ).mean()

    imp_reject=(
        si>=threshold
    ).mean()

    print()

    print("====== GENERALIZATION ======")

    print(
        f"Threshold: {threshold:.6f}"
    )

    print(
        f"Owner Accept: {owner_accept:.4f}"
    )

    print(
        f"Impostor Reject: {imp_reject:.4f}"
    )

    print(
        f"Owner Median: {np.median(so):.6f}"
    )

    print(
        f"Impostor Median: {np.median(si):.6f}"
    )

    separation=(
        np.median(si)
        /
        max(
            np.median(so),
            1e-8
        )
    )

    print(
        f"Separation: {separation:.2f}x"
    )

    Path(
        "evaluation"
    ).mkdir(
        exist_ok=True
    )

    json.dump(
        {
            "owner_accept":
            float(
                owner_accept
            ),

            "impostor_reject":
            float(
                imp_reject
            ),

            "separation":
            float(
                separation
            )
        },

        open(
            "evaluation/generalization.json",
            "w"
        ),

        indent=2
    )


if __name__=="__main__":
    main()
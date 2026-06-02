import pickle
import torch
import pandas as pd
import numpy as np

from model_architecture import BehavioralAutoencoder


MODEL="checkpoints/best_model.pth"
SCALER="checkpoints/scaler.pkl"

THRESHOLD=0.200663


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


model=BehavioralAutoencoder()

model.load_state_dict(
torch.load(
MODEL,
map_location="cpu"
)
)

model.eval()


scaler=pickle.load(
open(
SCALER,
"rb"
)
)


def authenticate(df):

    x=scaler.transform(
        df[
            FEATURES
        ]
    )

    x=torch.tensor(
        x,
        dtype=torch.float32
    )

    with torch.no_grad():

        recon=model(
            x
        )

        err=(

        (
        recon-x
        )**2

        ).mean(
        dim=1
        )

    score=float(
        err.mean()
    )

    return {

    "score":
    score,

    "verified":
    (
    score
    <
    THRESHOLD
    )

    }
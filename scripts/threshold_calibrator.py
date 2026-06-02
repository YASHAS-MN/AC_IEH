# scripts/threshold_calibrator.py

import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model_architecture import (
    create_model,
    load_checkpoint
)

DEVICE=(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR=Path(__file__).resolve().parents[1]

DATA=ROOT_DIR/"master_feature_dataset.parquet"

MODEL_PATH=ROOT_DIR/"checkpoints"/"best_model.pth"

SCALER_PATH=ROOT_DIR/"checkpoints"/"scaler.pkl"


###################################################
# Load dataset
###################################################

print("\nLoading dataset...")

df=pd.read_parquet(
DATA
)

labels=df[
"is_owner"
].values


drop_cols=[

"context",
"device",
"is_owner"

]

X=df.drop(
columns=drop_cols
)


###################################################
# Normalize
###################################################

scaler=joblib.load(
SCALER_PATH
)

X=scaler.transform(
X
)

X=torch.tensor(
X,
dtype=torch.float32
).to(
DEVICE
)


###################################################
# Load model
###################################################

input_dim=X.shape[1]

model=create_model(
input_dim
)

model=load_checkpoint(

model,
MODEL_PATH,
DEVICE

)

model=model.to(
DEVICE
)


###################################################
# Reconstruction error
###################################################

model.eval()

with torch.no_grad():

    reconstructed=model(
    X
    )

    errors=torch.mean(

        (X-reconstructed)**2,

        dim=1

    )

errors=errors.cpu().numpy()



###################################################
# ROC
###################################################

fpr,tpr,thresholds=roc_curve(

labels,
-errors

)


###################################################
# Equal Error Rate
###################################################

eer=brentq(

lambda x:
1.-x-interp1d(
fpr,
tpr
)(x),

0.,
1.

)

eer_threshold=interp1d(

fpr,
thresholds

)(eer)



###################################################
# FAR FRR
###################################################

pred=(
-errors
>=
eer_threshold
).astype(
int
)

far=np.mean(

(pred==1)
&
(labels==0)

)

frr=np.mean(

(pred==0)
&
(labels==1)

)


###################################################
# Report
###################################################

print("\n======================")

print(
f"EER: {eer:.4f}"
)

print(
f"Threshold: {float(eer_threshold):.4f}"
)

print(
f"FAR: {far:.4f}"
)

print(
f"FRR: {frr:.4f}"
)

print("======================")

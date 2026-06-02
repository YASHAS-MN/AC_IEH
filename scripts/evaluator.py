import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model_architecture import create_model


ROOT = Path(".")

DATA = ROOT / "master_feature_dataset.parquet"

MODEL = ROOT / "checkpoints" / "best_model.pth"
SCALER = ROOT / "checkpoints" / "scaler.pkl"

OUT = ROOT / "evaluation"

OUT.mkdir(exist_ok=True)


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
    "flight_std"
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reconstruction_error(
    model,
    x
):

    with torch.no_grad():

        pred = model(x)

        err = (
            (
                pred - x
            ) ** 2
        ).mean(
            dim=1
        )

    return (
        err
        .cpu()
        .numpy()
    )


print("\nLoading dataset...")

df = pd.read_parquet(DATA)

owner = (
    df[
        df.is_owner == 1
    ]
    .copy()
)

impostor = (
    df[
        df.is_owner == 0
    ]
    .copy()
)

print(
    f"\nOwner={len(owner)}"
)

print(
    f"Impostor={len(impostor)}"
)


scaler = (
    joblib
    .load(
        SCALER
    )
)

x_owner = scaler.transform(
    owner[
        FEATURES
    ]
)

x_imp = scaler.transform(
    impostor[
        FEATURES
    ]
)


model = create_model(
    len(FEATURES)
)

model.load_state_dict(

    torch.load(
        MODEL,
        map_location=DEVICE
    )

)

model.eval()

model.to(
    DEVICE
)


owner_tensor = torch.tensor(
    x_owner,
    dtype=torch.float32
).to(
    DEVICE
)

imp_tensor = torch.tensor(
    x_imp,
    dtype=torch.float32
).to(
    DEVICE
)


print(
    "\nScoring..."
)

owner_err = reconstruction_error(
    model,
    owner_tensor
)

imp_err = reconstruction_error(
    model,
    imp_tensor
)


threshold = np.percentile(
    owner_err,
    99
)


# Compute comprehensive statistics
owner_stats = {
    "mean": float(owner_err.mean()),
    "median": float(np.percentile(owner_err, 50)),
    "p90": float(np.percentile(owner_err, 90)),
    "p95": float(np.percentile(owner_err, 95)),
    "p99": float(np.percentile(owner_err, 99)),
    "max": float(owner_err.max()),
}

impostor_stats = {
    "mean": float(imp_err.mean()),
    "median": float(np.percentile(imp_err, 50)),
    "p90": float(np.percentile(imp_err, 90)),
    "p95": float(np.percentile(imp_err, 95)),
    "p99": float(np.percentile(imp_err, 99)),
    "max": float(imp_err.max()),
}

# ============================================
# Robust Outlier Detection (MAD Method)
# ============================================

def detect_outliers_mad(errors, multiplier=10):
    """Detect outliers using Median Absolute Deviation"""
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    upper_threshold = median + multiplier * mad
    outlier_mask = errors > upper_threshold
    return outlier_mask, median, mad, upper_threshold

# Owner outliers
owner_outlier_mask, owner_median, owner_mad, owner_upper = detect_outliers_mad(owner_err)
owner_outlier_count = int(owner_outlier_mask.sum())

# Impostor outliers
impostor_outlier_mask, impostor_median, impostor_mad, impostor_upper = detect_outliers_mad(imp_err)
impostor_outlier_count = int(impostor_outlier_mask.sum())

print(f"\nOutlier Detection (MAD Method):")
print(f"Owner outliers: {owner_outlier_count} (threshold: {owner_upper:.6f})")
print(f"Impostor outliers: {impostor_outlier_count} (threshold: {impostor_upper:.6f})")

# Clean data (remove outliers)
owner_err_clean = owner_err[~owner_outlier_mask]
impostor_err_clean = imp_err[~impostor_outlier_mask]

# Compute clean statistics
owner_stats_clean = {
    "median": float(np.percentile(owner_err_clean, 50)),
    "p90": float(np.percentile(owner_err_clean, 90)),
    "p95": float(np.percentile(owner_err_clean, 95)),
    "p99": float(np.percentile(owner_err_clean, 99)),
    "max": float(owner_err_clean.max()),
}

impostor_stats_clean = {
    "median": float(np.percentile(impostor_err_clean, 50)),
    "p90": float(np.percentile(impostor_err_clean, 90)),
    "p95": float(np.percentile(impostor_err_clean, 95)),
    "p99": float(np.percentile(impostor_err_clean, 99)),
    "max": float(impostor_err_clean.max()),
}

metrics = {
    "owner": owner_stats,
    "impostor": impostor_stats,
    "owner_outlier_count": owner_outlier_count,
    "impostor_outlier_count": impostor_outlier_count,
    "owner_mad_threshold": float(owner_upper),
    "impostor_mad_threshold": float(impostor_upper),
}

metrics_clean = {
    "owner": owner_stats_clean,
    "impostor": impostor_stats_clean,
    "owner_outlier_count": owner_outlier_count,
    "impostor_outlier_count": impostor_outlier_count,
}


with open(
OUT/"statistics.json",
"w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


with open(
OUT/"statistics_clean.json",
"w"
) as f:

    json.dump(
        metrics_clean,
        f,
        indent=4
    )


# Linear distribution histogram
plt.figure(
    figsize=(12, 7)
)

plt.hist(
    owner_err,
    bins=100,
    alpha=0.6,
    label="Owner"
)

plt.hist(
    imp_err,
    bins=100,
    alpha=0.6,
    label="Impostor"
)

plt.axvline(
    threshold,
    linewidth=3
)

plt.xlabel(
    "Reconstruction Error"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Error Distribution (Linear Scale)"
)

plt.legend()

plt.savefig(
    OUT/"distribution_linear.png",
    dpi=300
)

plt.close()


# Log-scale distribution histogram
plt.figure(
    figsize=(12, 7)
)

owner_err_log = np.log10(owner_err + 1)
imp_err_log = np.log10(imp_err + 1)

plt.hist(
    owner_err_log,
    bins=100,
    alpha=0.6,
    label="Owner"
)

plt.hist(
    imp_err_log,
    bins=100,
    alpha=0.6,
    label="Impostor"
)

threshold_log = np.log10(threshold + 1)
plt.axvline(
    threshold_log,
    linewidth=3
)

plt.xlabel(
    "log10(Reconstruction Error + 1)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Error Distribution (Log Scale)"
)

plt.legend()

plt.savefig(
    OUT/"distribution_log.png",
    dpi=300
)

plt.close()


# Clean distribution histogram (after outlier removal)
plt.figure(
    figsize=(12, 7)
)

plt.hist(
    owner_err_clean,
    bins=100,
    alpha=0.6,
    label="Owner (Clean)"
)

plt.hist(
    impostor_err_clean,
    bins=100,
    alpha=0.6,
    label="Impostor (Clean)"
)

plt.axvline(
    threshold,
    linewidth=3,
    label="Threshold"
)

plt.xlabel(
    "Reconstruction Error"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Error Distribution (After Outlier Removal)"
)

plt.legend()

plt.savefig(
    OUT/"distribution_clean.png",
    dpi=300
)

plt.close()


accept_owner = (
owner_err
<=
threshold
).mean()

reject_imp = (
imp_err
>
threshold
).mean()


print("\n==========")

print(
    f"Threshold: {threshold:.6f}"
)

print(
    f"Owner Accept: {accept_owner:.4f}"
)

print(
    f"Impostor Reject: {reject_imp:.4f}"
)

print("\n--- Clean Statistics ---")
print(
    f"Owner Median: {owner_stats_clean['median']:.6f}"
)

print(
    f"Impostor Median: {impostor_stats_clean['median']:.6f}"
)

print(
    f"Owner P95: {owner_stats_clean['p95']:.6f}"
)

print(
    f"Impostor P95: {impostor_stats_clean['p95']:.6f}"
)

print(
    f"Impostor Outliers: {impostor_outlier_count}"
)

print("==========")

print(
    "\nSaved:"
)

print(
    OUT/"distribution_linear.png"
)

print(
    OUT/"distribution_log.png"
)

print(
    OUT/"distribution_clean.png"
)

print(
    OUT/"statistics.json"
)

print(
    OUT/"statistics_clean.json"
)

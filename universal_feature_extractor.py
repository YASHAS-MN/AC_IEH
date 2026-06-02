# universal_feature_extractor.py

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from fastparquet import write

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

ROOT=Path(".")

TRUE_DIR=ROOT/"yashas_data"
NEG_DIR=ROOT/"imposters"

OUTPUT=ROOT/"master_feature_dataset.parquet"

WINDOW_SIZE=50
STRIDE=25
RESAMPLE_MS=10

DEFAULT_W=1920
DEFAULT_H=1080


####################################################
# Utility
####################################################

def entropy(x):

    x=np.asarray(x,dtype=float)

    x=x[np.isfinite(x)]

    if x.size<2:
        return 0.0

    x_min=x.min()
    x_max=x.max()

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return 0.0

    if np.isclose(x_min,x_max):
        return 0.0

    hist=np.histogram(
        x,
        bins=min(20,x.size),
        range=(x_min,x_max)
    )[0]

    p=hist/(hist.sum()+1e-8)

    p=p[p>0]

    return -np.sum(
        p*np.log2(p)
    )


def normalize_xy(x,y,w,h):

    return x/w,y/h


####################################################
# Schema Detection
####################################################

def detect_schema(df):

    cols=set(df.columns)

    if "vector_param_0" in cols:
        return "yashas"

    elif {"x","y","event_type"}<=cols:
        return "generic_mouse"

    elif {"dwell_ms","flight_ms"}<=cols:
        return "keystroke"

    elif {"x_norm","y_norm"}<=cols:
        return "normalized_mouse"

    return "unknown"


####################################################
# Adapters
####################################################

def adapt(df,schema):


    #########################################
    # Yashas raw
    #########################################

    if schema=="yashas":

        if "move" in df["event_class"].values:

            out=pd.DataFrame()

            out["timestamp"]=df["timestamp"]

            out["x_norm"]=(
                df["vector_param_0"]
                /
                DEFAULT_W
            )

            out["y_norm"]=(
                df["vector_param_1"]
                /
                DEFAULT_H
            )

            out["app_context"]="unknown"

            out["modality"]="mouse"

            return out


        else:

            out=pd.DataFrame()

            out["timestamp"]=df["timestamp"]

            keyup=df[
                df["event_class"]=="keyup"
            ].copy()

            out["dwell_ms"]=(
                keyup["vector_param_1"]
            )

            out["flight_ms"]=(
                out["dwell_ms"]
                .diff()
                .fillna(0)
            )

            out["app_context"]="unknown"

            out["modality"]="keyboard"

            return out



    #########################################
    # Dell/Fira style
    #########################################

    elif schema=="generic_mouse":

        if "event_type" in df.columns:

            events=(
                df["event_type"]
                .astype(str)
                .str.lower()
            )

            df=df[

                events.isin([

                    "move",
                    "mouse_move",

                    "click",
                    "click_down",
                    "click_up"

                ])

            ].copy()

        if df.empty:
            return pd.DataFrame()

        out=pd.DataFrame()

        out["timestamp"]=df["timestamp"]

        if "resolution" in df.columns:

            split=df[
                "resolution"
            ].str.split(
                "x",
                expand=True
            )

            w=split[0].astype(float)

            h=split[1].astype(float)

        else:

            w=DEFAULT_W
            h=DEFAULT_H


        out["x_norm"]=df["x"]/w

        out["y_norm"]=df["y"]/h

        out["app_context"]=df.get(
            "context",
            "unknown"
        )

        out["modality"]="mouse"

        return out



    #########################################

    elif schema=="normalized_mouse":

        out=pd.DataFrame()

        out["timestamp"]=df[
            "timestamp"
        ]

        out["x_norm"]=df[
            "x_norm"
        ]

        out["y_norm"]=df[
            "y_norm"
        ]

        out["app_context"]=df.get(
            "app_context",
            "unknown"
        )

        out["modality"]="mouse"

        return out



    #########################################

    elif schema=="keystroke":

        out=pd.DataFrame()

        out["timestamp"]=df[
            "timestamp"
        ]

        out["dwell_ms"]=df[
            "dwell_ms"
        ]

        out["flight_ms"]=df[
            "flight_ms"
        ].fillna(
            0
        )

        out = out[
            (out["flight_ms"] >= 0)
            &
            (out["flight_ms"] <= 3000)
        ]

        out["app_context"]=df.get(
            "app_context",
            "unknown"
        )

        out["modality"]="keyboard"

        return out


    return pd.DataFrame()



####################################################
# Mouse Features
####################################################

def mouse_features(df):

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce"
    )

    df = df.loc[df["timestamp"].notna()].copy()

    df = df.sort_values(
        "timestamp"
    )

    dt = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df = df.loc[dt > 0].copy()
    dt = dt.loc[dt > 0]

    # ============================================
    # 1. Minimum dt floor
    # ============================================
    dt_floor = 0.005
    clamped_mask = dt < dt_floor
    clamped_count = clamped_mask.sum()
    if clamped_count > 0:
        logging.info(f"Clamped dt rows: {clamped_count}")
        dt = dt.clip(lower=dt_floor)

    dx = df["x_norm"].diff()
    dy = df["y_norm"].diff()

    # ============================================
    # 2. Stable derivatives with clamped dt
    # ============================================
    df["velocity"]=(
        np.sqrt(
            dx**2+dy**2
        )
        /
        dt
    )
    df["velocity"] = df["velocity"].replace([np.inf, -np.inf], np.nan)
    df["velocity"] = df["velocity"].fillna(0)

    df["acceleration"]=(
        df["velocity"]
        .diff()
        /
        dt
    )
    df["acceleration"] = df["acceleration"].replace([np.inf, -np.inf], np.nan)
    df["acceleration"] = df["acceleration"].fillna(0)

    df["jerk"]=(
        df["acceleration"]
        .diff()
        /
        dt
    )
    df["jerk"] = df["jerk"].replace([np.inf, -np.inf], np.nan)
    df["jerk"] = df["jerk"].fillna(0)

    d2x=dx.diff()
    d2y=dy.diff()

    numerator=(
        dx*d2y
        -
        dy*d2x
    ).abs()

    denominator=(
        (dx**2+dy**2)**1.5
    )+1e-8

    df["curvature"]=(
        numerator
        /
        denominator
    )
    df["curvature"] = df["curvature"].replace([np.inf, -np.inf], np.nan)
    df["curvature"] = df["curvature"].fillna(0)

    # ============================================
    # 3. Percentile clipping
    # ============================================
    velocity_lower = df["velocity"].quantile(0.005)
    velocity_upper = df["velocity"].quantile(0.995)
    velocity_before_max = df["velocity"].max()
    df["velocity"] = df["velocity"].clip(lower=velocity_lower, upper=velocity_upper)
    velocity_clipped = (df["velocity"] != velocity_before_max).sum()
    if velocity_clipped > 0:
        logging.info(f"velocity clipped: {velocity_clipped}")

    acceleration_lower = df["acceleration"].quantile(0.01)
    acceleration_upper = df["acceleration"].quantile(0.99)
    acceleration_before_max = df["acceleration"].max()
    df["acceleration"] = df["acceleration"].clip(lower=acceleration_lower, upper=acceleration_upper)
    acceleration_clipped = (df["acceleration"] != acceleration_before_max).sum()
    if acceleration_clipped > 0:
        logging.info(f"acceleration clipped: {acceleration_clipped}")

    jerk_lower = df["jerk"].quantile(0.01)
    jerk_upper = df["jerk"].quantile(0.99)
    jerk_before_max = df["jerk"].max()
    df["jerk"] = df["jerk"].clip(lower=jerk_lower, upper=jerk_upper)
    jerk_clipped = (df["jerk"] != jerk_before_max).sum()
    if jerk_clipped > 0:
        logging.info(f"jerk clipped: {jerk_clipped}")

    curvature_lower = df["curvature"].quantile(0.01)
    curvature_upper = df["curvature"].quantile(0.99)
    curvature_before_max = df["curvature"].max()
    df["curvature"] = df["curvature"].clip(lower=curvature_lower, upper=curvature_upper)
    curvature_clipped = (df["curvature"] != curvature_before_max).sum()
    if curvature_clipped > 0:
        logging.info(f"curvature clipped: {curvature_clipped}")

    # ============================================
    # 4. Diagnostics
    # ============================================
    logging.info(f"velocity max: {df['velocity'].max():.6f}")
    logging.info(f"acceleration max: {df['acceleration'].max():.6f}")
    logging.info(f"jerk max: {df['jerk'].max():.6f}")

    return df



####################################################
# Window Aggregation
####################################################

def windows(df,label,device):

    rows=[]

    is_keyboard=(
        "dwell_ms" in df.columns
        or
        "flight_ms" in df.columns
    )

    if is_keyboard:

        local_window=15
        local_stride=5

    else:

        local_window=50
        local_stride=25


    for i in range(

        0,
        len(df)-local_window+1,
        local_stride

    ):

        w=df.iloc[
            i:i+local_window
        ].copy()

        if len(w)<local_window:
            continue


        for col in [

            "velocity",
            "acceleration",
            "jerk",
            "curvature"

        ]:

            if col not in w.columns:

                w.loc[:,col]=0.0

        row={

        "velocity_mean":
        w["velocity"].mean(),

        "velocity_std":
        w["velocity"].std(),

        "velocity_max":
        w["velocity"].max(),

        "acceleration_mean":
        w["acceleration"].mean(),

        "acceleration_std":
        w["acceleration"].std(),

        "jerk_mean":
        w["jerk"].mean(),

        "jerk_std":
        w["jerk"].std(),

        "curvature_mean":
        w["curvature"].mean(),

        "curvature_std":
        w["curvature"].std(),

        "trajectory_entropy":
        entropy(
        w["velocity"].values
        ),

        "dwell_mean":
        w.get(
        "dwell_ms",
        pd.Series([0])
        ).mean(),

        "dwell_std":
        w.get(
        "dwell_ms",
        pd.Series([0])
        ).std(skipna=True),

        "flight_mean":
        w.get(
        "flight_ms",
        pd.Series([0])
        ).mean(),

        "flight_std":
        w.get(
        "flight_ms",
        pd.Series([0])
        ).std(skipna=True),

        "context":
        w.get(
        "app_context",
        pd.Series(
        ["unknown"]
        )
        ).mode().iloc[0],

        "device":
        device,

        "is_owner":
        label
        }

        rows.append(
            row
        )

    return rows



####################################################
# Main
####################################################

def main():

    all_rows=[]


    for label,path in [

    (1,TRUE_DIR),
    (0,NEG_DIR)

    ]:

        files=list(
            path.rglob(
            "*.parquet"
            )
        )

        for file in files:

            logging.info(
            f"Processing {file.name}"
            )

            df=pd.read_parquet(
                file
            )

            schema=detect_schema(
                df
            )

            logging.info(
            f"Schema: {schema}"
            )

            adapted=adapt(
                df,
                schema
            )

            if adapted.empty:
                continue

            device=file.stem

            if (
            adapted["modality"]
            .iloc[0]
            ==
            "mouse"
            ):

                adapted=mouse_features(
                    adapted
                )

            rows=windows(
                adapted,
                label,
                device
            )

            all_rows.extend(
                rows
            )


    final=pd.DataFrame(
    all_rows
    )

    final=final.fillna(0)

    write(
    OUTPUT,
    final,
    compression="SNAPPY",
    write_index=False
    )

    print(
    "\nDone"
    )

    print(
    final["is_owner"]
    .value_counts()
    )

    print(
    final["device"]
    .value_counts()
    )


if __name__=="__main__":
    main()

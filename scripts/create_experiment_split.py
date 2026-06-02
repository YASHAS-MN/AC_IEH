from pathlib import Path
import pandas as pd

DATA="master_feature_dataset.parquet"

OUT="experiment"

Path(OUT).mkdir(
    exist_ok=True
)

df=pd.read_parquet(DATA)

owner=df[
    df.is_owner==1
].copy()

imp=df[
    df.is_owner==0
].copy()

owner=owner.reset_index(
    drop=True
)

split=int(
    len(owner)*0.7
)

owner_train=owner.iloc[:split]

owner_test=owner.iloc[split:]

owner_train.to_parquet(
    f"{OUT}/owner_train.parquet",
    index=False
)

owner_test.to_parquet(
    f"{OUT}/owner_test.parquet",
    index=False
)

imp.to_parquet(
    f"{OUT}/impostor_test.parquet",
    index=False
)

print()

print(
"Owner Train:",
len(owner_train)
)

print(
"Owner Test:",
len(owner_test)
)

print(
"Impostor Test:",
len(imp)
)
# scripts/trainer_engine.py

import os
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_architecture import (
    create_model,
    save_checkpoint
)

DEVICE=(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

ROOT_DIR=Path(__file__).resolve().parents[1]
FEATURE_FILE=ROOT_DIR/"master_feature_dataset.parquet"

BATCH_SIZE=64
LEARNING_RATE=1e-3
EPOCHS=100
PATIENCE=10

CHECKPOINT_DIR=ROOT_DIR/"checkpoints"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


class BehaviorDataset(Dataset):

    def __init__(self,data):

        self.x=torch.tensor(
            data,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.x)

    def __getitem__(self,idx):

        return self.x[idx]



print("\nLoading dataset...")

df=pd.read_parquet(
    FEATURE_FILE
)


##################################################
# owner only training
##################################################

df=df[
    df["is_owner"]==1
]


drop_cols=[

"context",
"device",
"is_owner"

]

X=df.drop(
    columns=drop_cols
)

print(
f"Owner samples: {len(X)}"
)


##################################################
# normalize
##################################################

scaler=StandardScaler()

X=scaler.fit_transform(
    X
)

joblib.dump(
    scaler,
    CHECKPOINT_DIR/"scaler.pkl"
)


##################################################
# train validation split
##################################################

X_train,X_val=\
train_test_split(

X,

test_size=0.2,

random_state=42

)


train_loader=DataLoader(

BehaviorDataset(
X_train
),

batch_size=BATCH_SIZE,
shuffle=True

)


val_loader=DataLoader(

BehaviorDataset(
X_val
),

batch_size=BATCH_SIZE

)



##################################################
# model
##################################################

input_dim=X.shape[1]

model=create_model(
input_dim
)

model=model.to(
DEVICE
)

criterion=torch.nn.MSELoss()

optimizer=torch.optim.Adam(

model.parameters(),
lr=LEARNING_RATE

)



##################################################
# train loop
##################################################

best_loss=np.inf

patience_counter=0


for epoch in range(
EPOCHS
):

    model.train()

    train_loss=0


    for batch in train_loader:

        batch=batch.to(
        DEVICE
        )

        reconstructed=\
        model(
        batch
        )

        loss=criterion(

        reconstructed,
        batch

        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss+=\
        loss.item()



    #################################################

    model.eval()

    val_loss=0

    with torch.no_grad():

        for batch in val_loader:

            batch=batch.to(
            DEVICE
            )

            reconstructed=\
            model(
            batch
            )

            loss=criterion(

            reconstructed,
            batch

            )

            val_loss+=\
            loss.item()


    train_loss/=len(
    train_loader
    )

    val_loss/=len(
    val_loader
    )


    print(

    f"Epoch {epoch+1}"

    f" | Train:{train_loss:.4f}"

    f" | Val:{val_loss:.4f}"

    )


    #########################################

    if val_loss<best_loss:

        best_loss=val_loss

        patience_counter=0

        save_checkpoint(

        model,

        os.path.join(

        CHECKPOINT_DIR,

        "best_model.pth"

        )

        )

    else:

        patience_counter+=1


    if patience_counter>=PATIENCE:

        print(
        "\nEarly stopping"
        )

        break


print(
f"\nBest validation loss:{best_loss:.6f}"
)

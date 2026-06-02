# scripts/model_architecture.py

import torch
import torch.nn as nn


class BehavioralAutoencoder(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Linear(input_dim,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,8)

        )

        self.decoder=nn.Sequential(

            nn.Linear(8,16),
            nn.ReLU(),

            nn.Linear(16,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,input_dim)

        )


    def forward(self,x):

        latent=self.encoder(x)

        reconstructed=self.decoder(latent)

        return reconstructed



def create_model(input_dim):

    model=BehavioralAutoencoder(
        input_dim=input_dim
    )

    return model



def save_checkpoint(
        model,
        path
):

    torch.save(

        model.state_dict(),
        path

    )



def load_checkpoint(
        model,
        path,
        device="cpu"
):

    model.load_state_dict(

        torch.load(
            path,
            map_location=device
        )

    )

    model.eval()

    return model



def export_onnx(
        model,
        input_dim,
        path
):

    dummy=torch.randn(
        1,
        input_dim
    )

    torch.onnx.export(

        model,

        dummy,

        path,

        export_params=True,

        opset_version=11,

        input_names=[
            "behavior_input"
        ],

        output_names=[
            "reconstruction"
        ],

        dynamic_axes={

            "behavior_input":{

                0:"batch"

            }

        }

    )



if __name__=="__main__":

    INPUT_DIM=15

    model=create_model(
        INPUT_DIM
    )

    print(model)

    total_params=sum(

        p.numel()

        for p in model.parameters()

    )

    print(
        f"\nParameters: {total_params:,}"
    )
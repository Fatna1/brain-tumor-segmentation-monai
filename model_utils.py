# model_utils.py
import torch
import os
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

import config


def build_model(device):
    model = UNet(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        channels=config.CHANNELS,
        strides=config.STRIDES,
        num_res_units=config.NUM_RES_UNITS,
    ).to(device)
    return model


def build_loss():
    return DiceCELoss(to_onehot_y=True, softmax=True)


def build_optimizer(model, lr=config.LR):
    return torch.optim.Adam(model.parameters(), lr)


def save_model(model, path=config.MODEL_SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model(model, path=config.MODEL_SAVE_PATH, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {path}")
    return model
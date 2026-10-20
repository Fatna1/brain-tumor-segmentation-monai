import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

import config
from data_utils import get_datasets
from model_utils import build_model, load_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  weights: {config.MODEL_SAVE_PATH}")
    print(f"Weights exist: {os.path.isfile(config.MODEL_SAVE_PATH)}")

    _, val_ds = get_datasets()
    model = build_model(device)
    load_model(model, config.MODEL_SAVE_PATH, device)

    # pick first volume that has tumor
    idx = next(i for i in range(30) if (val_ds[i]["label"][0] > 0).sum() > 1000)
    sample = val_ds[idx]
    img = sample["image"].unsqueeze(0).to(device)
    gt = sample["label"].numpy().squeeze(0)                     # (D,H,W)

    with torch.no_grad():
        logits = sliding_window_inference(img, config.SPATIAL_SIZE, 4, model)
    pred = torch.argmax(logits, 1).squeeze(0).cpu().numpy()      # (D,H,W)

    # slice with most tumor
    s = int((gt > 0).sum(axis=(0, 1)).argmax())
    bg = img.squeeze(0)[0, :, :, s].cpu().numpy()

    # quick Dice
    dm = DiceMetric(include_background=False)
    post = AsDiscrete(to_onehot=config.OUT_CHANNELS)
    dm(y_pred=[post(torch.from_numpy(pred).unsqueeze(0))],
       y=[post(torch.from_numpy(gt).unsqueeze(0))])
    dice = dm.aggregate().item()

    # 3-panel figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    for a, title, mask in [(ax[0], "MRI", None),
                           (ax[1], "Ground truth", gt[:, :, s]),
                           (ax[2], f"Prediction (Dice {dice:.3f})", pred[:, :, s])]:
        a.imshow(bg, cmap="gray"); a.set_title(title); a.axis("off")
        if mask is not None:
            a.imshow(np.ma.masked_where(mask == 0, mask),
                     cmap="jet", alpha=0.5, vmin=1, vmax=3)

    out = "/com_neuro_project/weights/viz.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Volume Dice: {dice:.4f}")


if __name__ == "__main__":
    main()
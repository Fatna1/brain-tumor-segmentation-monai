
import os
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import json
import config
from data_utils import get_dataloaders
from model_utils import build_model, load_model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # get_dataloaders() returns both; we ignore train_loader
    _, val_loader = get_dataloaders()
    print(f"Validation batches: {len(val_loader)}")

    # Confirm weights are visible before we waste T4 time
    print(f"Loading weights from: {config.MODEL_SAVE_PATH}")
    print(f"Weights file exists: {os.path.isfile(config.MODEL_SAVE_PATH)}")

    model = build_model(device)
    load_model(model, config.MODEL_SAVE_PATH, device)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean_batch"
    )
    post_trans = AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=config.OUT_CHANNELS)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)

            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=config.SPATIAL_SIZE,
                sw_batch_size=4,
                predictor=model,
            )

            val_outputs_list = [post_trans(t) for t in decollate_batch(val_outputs)]
            val_labels_list = [post_label(t) for t in decollate_batch(val_labels)]

            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            hd_metric(y_pred=val_outputs_list, y=val_labels_list)

            if (i + 1) % 10 == 0:
                print(f"  processed {i + 1}/{len(val_loader)} volumes")


    # NEW — handle per-class + overall mean
    per_class_dice = dice_metric.aggregate()      # tensor shape (3,)
    per_class_hd = hd_metric.aggregate()          # tensor shape (3,)

    mean_dice = per_class_dice.mean().item()
    mean_hd = per_class_hd.mean().item()

    print(f"Per-class Dice: {per_class_dice.tolist()}")
    print(f"Per-class HD95: {per_class_hd.tolist()}")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean Hausdorff Distance (HD95): {mean_hd:.4f} voxels")

    dice_metric.reset()
    hd_metric.reset()
   
    return mean_dice, mean_hd
    results_path = "/com_neuro_project/weights/eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
        "mean_dice": float(mean_dice),
        "hd95": float(mean_hd),
        "per_class_dice": per_class_dice.tolist() if 'per_class_dice' in dir() else None,
        }, f, indent=2) 
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    evaluate()
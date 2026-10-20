import torch
import config
from data_utils import get_dataloaders
from model_utils import build_model, build_loss, build_optimizer, save_model
import os
print(f"Will save weights to: {config.MODEL_SAVE_PATH}")
print(f"Parent exists: {os.path.isdir(os.path.dirname(config.MODEL_SAVE_PATH))}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, _ = get_dataloaders()
    print(f"Training batches: {len(train_loader)}")

    model = build_model(device)
    loss_fct = build_loss()
    optimizer = build_optimizer(model)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fct(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {epoch_loss/step:.4f}")

    print("Training phase is done")
    save_model(model)


if __name__ == "__main__":
    train()

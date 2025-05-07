# train.py
import torch
from tqdm import tqdm
# Import Automatic Mixed Precision (AMP) modules for efficient training.
from torch.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, optimizer, device, lr_scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training data.
        optimizer: The optimization algorithm.
        device: The device to train on (e.g., 'cuda' or 'cpu').
        lr_scheduler: Optional learning rate scheduler.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    scaler = GradScaler(enabled=(device.type == 'cuda'))
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            loss_dict = model(images, targets)
            loss = sum(loss_val for loss_val in loss_dict.values())

        total_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None and \
           not isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if lr_scheduler is not None and \
       isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
        lr_scheduler.step()

    return total_loss / num_batches if num_batches > 0 else 0.0

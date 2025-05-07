import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, optimizer, device, lr_scheduler=None):
    """Trains model for one epoch with AMP and optional LR scheduler."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    scaler = GradScaler('cuda')

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast('cuda'):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        total_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None and not isinstance(
                lr_scheduler, torch.optim.lr_scheduler.StepLR):
            lr_scheduler.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if lr_scheduler is not None and isinstance(
            lr_scheduler, torch.optim.lr_scheduler.StepLR):
        lr_scheduler.step()

    return total_loss / num_batches if num_batches > 0 else 0

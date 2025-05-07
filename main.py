import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch.optim.lr_scheduler as lr_scheduler # Import LR schedulers
import csv # Import the csv module

# Import modules
from config import get_config
from utils import set_seed, collate_fn
from data import InstanceSegmentTrainDataset, create_gt_coco, get_transform
from models import MaskRCNN, MaskRCNN_v2, MaskRCNN_ConvNeXt, MaskRCNN_Cell # Import all model classes
from train import train_one_epoch
from eval import evaluate_model
from infer import inference

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main():
    # 1. Configuration and Setup
    cfg = get_config()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 2. Data Loading and Preparation
    dataset = InstanceSegmentTrainDataset(cfg.root_dir, transforms=get_transform(train=True))

    total_size = len(dataset)
    valid_size = int(cfg.valid_split * total_size)
    train_size = total_size - valid_size

    train_set, valid_set = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    valid_set.dataset.transforms = get_transform(train=False)

    print(f"Dataset split: {len(train_set)} training, {len(valid_set)} validation samples.")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # 3. Model Initialization based on config
    print(f"\nInitializing model: {cfg.model_arch}")
    if cfg.model_arch == 'MaskRCNN':
        model = MaskRCNN(num_classes=cfg.num_classes, pretrained=cfg.pretrained,
                         rpn_post_nms_train=cfg.rpn_post_nms_train,
                         rpn_post_nms_test=cfg.rpn_post_nms_test,
                         box_detections_per_img=cfg.box_detections_per_img).to(device)
    elif cfg.model_arch == 'MaskRCNN_v2':
        model = MaskRCNN_v2(num_classes=cfg.num_classes, pretrained=cfg.pretrained,
                            rpn_post_nms_train=cfg.rpn_post_nms_train,
                            rpn_post_nms_test=cfg.rpn_post_nms_test,
                            box_detections_per_img=cfg.box_detections_per_img).to(device)
    elif cfg.model_arch == 'MaskRCNN_ConvNeXt':
         model = MaskRCNN_ConvNeXt(num_classes=cfg.num_classes, pretrained=cfg.pretrained,
                                   rpn_post_nms_train=cfg.rpn_post_nms_train,
                                   rpn_post_nms_test=cfg.rpn_post_nms_test,
                                   box_detections_per_img=cfg.box_detections_per_img).to(device)
    elif cfg.model_arch == 'MaskRCNN_Cell':
         model = MaskRCNN_Cell(num_classes=cfg.num_classes, pretrained=cfg.pretrained,
                               rpn_post_nms_train=cfg.rpn_post_nms_train,
                               rpn_post_nms_test=cfg.rpn_post_nms_test,
                               box_detections_per_img=cfg.box_detections_per_img).to(device)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model_arch}")

    model.get_parameter_size()

    # 4. Optimizer
    optimizer = model.get_optimizer(base_lr=cfg.base_lr, weight_decay=cfg.weight_decay)

    # 5. LR Scheduler
    lr_scheduler_obj = None
    if cfg.lr_scheduler == 'step':
        lr_scheduler_obj = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
        print(f"Using StepLR scheduler with step_size={cfg.lr_step_size}, gamma={cfg.lr_gamma}")
    elif cfg.lr_scheduler == 'cosine':
        # T_max is typically the number of training iterations or epochs
        # For per-epoch stepping, T_max is num_epochs
        lr_scheduler_obj = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        print(f"Using CosineAnnealingLR scheduler with T_max={cfg.num_epochs}")
    elif cfg.lr_scheduler == 'none':
        print("No LR scheduler is used.")

    # 6. Prepare Ground Truth for Validation mAP
    print("Creating COCO ground truth for validation set...")
    gt_coco_dict = create_gt_coco(valid_set)
    gt_json_path = os.path.join(cfg.out_dir, "valid_gt.json")
    coco_gt = None

    with open(gt_json_path, "w") as f:
        json.dump(gt_coco_dict, f, indent=4)
    coco_gt = COCO(gt_json_path)
    print(f"Validation ground truth saved to {gt_json_path}")

    # 7. Training Loop
    best_valid_mAP = -1.0
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    valid_mAPs = []
    best_epoch = -1

    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{cfg.num_epochs} ---")

        # Pass the scheduler to the training function
        train_loss = train_one_epoch(model, train_loader, optimizer, device, lr_scheduler_obj)
        train_losses.append(train_loss)

        valid_loss, valid_mAP = evaluate_model(model, valid_loader, device, coco_gt)
        valid_losses.append(valid_loss)
        valid_mAPs.append(valid_mAP if valid_mAP is not None else -1)

        loss_str = f"{valid_loss:.4f}" if valid_loss is not None else "N/A"
        mAP_str = f"{valid_mAP:.4f}" if valid_mAP is not None else "N/A"
        print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f} | Valid Loss: {loss_str} | Valid mAP: {mAP_str}")

        save_criterion_met = False
        current_metric = -1.0

        if valid_mAP is not None and coco_gt is not None:
            current_metric = valid_mAP
            if current_metric > best_valid_mAP:
                print(f"Validation mAP improved ({best_valid_mAP:.4f} --> {current_metric:.4f}). Saving model...")
                best_valid_mAP = current_metric
                best_epoch = epoch + 1
                save_criterion_met = True
        elif valid_loss is not None:
            current_metric = valid_loss
            if current_metric < best_valid_loss:
                print(f"Validation Loss improved ({best_valid_loss:.4f} --> {current_metric:.4f}). Saving model...")
                best_valid_loss = current_metric
                best_epoch = epoch + 1
                save_criterion_met = True

        if save_criterion_met:
            best_model_path = os.path.join(cfg.out_dir, "best_model.pth")
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")
            except Exception:
                pass # Suppress save error print

    print(f"\nTraining finished. Best model from epoch {best_epoch} saved.")
    print(f"Best Validation mAP: {best_valid_mAP:.4f}" if best_valid_mAP != -1.0 else "N/A")
    print(f"Best Validation Loss: {best_valid_loss:.4f}" if best_valid_loss != float('inf') else "N/A")

    # 8. Save metrics to CSV file
    metrics_csv_path = os.path.join(cfg.out_dir, "training_metrics.csv")
    try:
        with open(metrics_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Valid mAP'])
            # Write data rows
            for i in range(cfg.num_epochs):
                epoch_num = i + 1
                train_loss = train_losses[i] if i < len(train_losses) else 'N/A'
                valid_loss = valid_losses[i] if i < len(valid_losses) and valid_losses[i] is not None else 'N/A'
                valid_mAP = valid_mAPs[i] if i < len(valid_mAPs) and valid_mAPs[i] != -1 else 'N/A'
                csv_writer.writerow([epoch_num, train_loss, valid_loss, valid_mAP])
        print(f"Training metrics saved to {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")


    # 9. Plotting Metrics
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, cfg.num_epochs + 1), train_losses, label='Train Loss')
        if all(v is not None for v in valid_losses):
            plt.plot(range(1, cfg.num_epochs + 1), valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        valid_mAPs_plot = [m if m is not None and m != -1 else np.nan for m in valid_mAPs]
        if not all(np.isnan(m) for m in valid_mAPs_plot):
            plt.plot(range(1, cfg.num_epochs + 1), valid_mAPs_plot, label='Valid mAP', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('mAP @ IoU=0.5:0.95')
            plt.legend()
            plt.title('Validation mAP')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'mAP not calculated', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Validation mAP (Not Available)')

        plt.tight_layout()
        metrics_plot_path = os.path.join(cfg.out_dir, "training_metrics.png")
        plt.savefig(metrics_plot_path)
        print(f"Training metrics plot saved to {metrics_plot_path}")
        plt.close()
    except Exception:
        pass # Suppress plotting error

    # 10. Inference on Test Set
    print("\nStarting inference on test set using best model...")
    best_model_path = os.path.join(cfg.out_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model weights from {best_model_path}")

        inference_output_path = os.path.join(cfg.out_dir, "test-results.json")
        inference(model, cfg.test_dir, cfg.test_json, device, output_json_path=inference_output_path)
    else:
        print(f"Warning: Best model file not found at {best_model_path}. Skipping inference.")


    print("\nScript finished.")

if __name__ == '__main__':
    main()

import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from torch.amp import autocast

from utils import encode_mask


def evaluate_model(model, dataloader, device, coco_gt=None):
    """Evaluates model, calculates loss and optional mAP with AMP."""
    model.eval()

    # Part 1: Calculate Loss (Optional)
    total_loss = 0.0
    num_batches_loss = 0

    model.train()
    with torch.no_grad():
        loss_progress_bar = tqdm(
            dataloader, desc="Calculating Eval Loss", leave=False
        )
        for images, targs in loss_progress_bar:
            images = [img.to(device) for img in images]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targs]

            with autocast('cuda'):
                try:
                    loss_dict = model(images, targs)
                    loss = sum(loss for loss in loss_dict.values())
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_batches_loss += 1
                except Exception:
                    total_loss = -1
                    break
    avg_loss = (
        total_loss / num_batches_loss if num_batches_loss > 0 and
        total_loss != -1 else None
    )

    model.eval()

    # Part 2: Collect Predictions for mAP
    predictions_coco = []
    if coco_gt is not None:
        pred_progress_bar = tqdm(
            dataloader, desc="Generating Predictions", leave=False
        )
        with torch.no_grad():
            for images, targets in pred_progress_bar:
                images = [img.to(device) for img in images]

                with autocast('cuda'):
                    outputs = model(images)

                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()

                    boxes = output["boxes"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    masks = output["masks"].cpu().numpy()

                    for j in range(len(scores)):
                        score = scores[j]
                        if score < 0.5:
                            continue

                        box = boxes[j].tolist()
                        xmin, ymin, xmax, ymax = box
                        bbox_coco = [xmin, ymin, xmax - xmin, ymax - ymin]
                        label = labels[j]
                        mask_binary = (masks[j, 0] > 0.5)
                        rle = encode_mask(mask_binary)

                        predictions_coco.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": bbox_coco,
                            "score": float(score),
                            "segmentation": rle,
                        })

    # Part 3: Compute mAP using COCOeval
    mAP = None
    if coco_gt is not None and predictions_coco:
        try:
            coco_dt = coco_gt.loadRes(predictions_coco)
            coco_eval = COCOeval(coco_gt, coco_dt, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
        except Exception:
            mAP = None

    elif coco_gt is not None and not predictions_coco:
        mAP = 0.0

    return avg_loss, mAP

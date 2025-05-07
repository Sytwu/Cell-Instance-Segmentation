import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F
from torch.amp import autocast

from utils import encode_mask
from models import MaskRCNN


def inference(model, test_dir, test_json_path, device,
              output_json_path="test-results.json", score_threshold=0.0):
    """Runs inference and saves predictions in COCO format with AMP."""
    with open(test_json_path) as f:
        image_info_list = json.load(f)

    model.eval()
    results_coco = []

    progress_bar = tqdm(image_info_list, desc="Inference")
    for info in progress_bar:
        file_name = info.get('file_name')
        image_id = info.get('id')

        file_path = os.path.join(test_dir, file_name)
        image = Image.open(file_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            with autocast('cuda'):
                outputs = model([image_tensor])
            output = outputs[0]

        scores = output['scores'].cpu().numpy()
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        masks = output['masks'].cpu().numpy()

        # Process detections
        for i in range(len(scores)):
            score = scores[i]
            if score < score_threshold:
                continue

            box = boxes[i].tolist()
            xmin, ymin, xmax, ymax = box
            width = max(0, xmax - xmin)
            height = max(0, ymax - ymin)
            bbox_coco = [xmin, ymin, width, height]

            label = labels[i]
            mask_binary = (masks[i, 0] > 0.5)
            rle = encode_mask(mask_binary)
            segm = rle

            results_coco.append({
                "image_id": image_id,
                "bbox": bbox_coco,
                "score": round(float(score), 5),
                "category_id": int(label),
                "segmentation": segm,
            })

    # Save results
    with open(output_json_path, 'w') as f:
        json.dump(results_coco, f, indent=4)
    print(f"Inference results saved to {output_json_path}")


if __name__ == "__main__":
    model_weights = 'best_model.pth'
    test_dir = 'hw3-data-release/test_release'
    test_json_path = 'hw3-data-release/test_image_name_to_ids.json'
    device = 'cuda'

    model = MaskRCNN(box_detections_per_img=300).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))

    inference(model, test_dir, test_json_path, device)

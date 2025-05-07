import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import read_maskfile, encode_mask


def get_transform(train=True):
    """Returns data augmentation pipeline."""
    transforms = []
    if train:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
        ])
    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids']
        ),
        additional_targets={'mask': 'mask'}
    )


class InstanceSegmentTrainDataset(Dataset):
    """Dataset for instance segmentation training."""
    def __init__(self, root_dir, transforms=None):
        """Initializes dataset."""
        self.root_dir = root_dir
        self.transforms = transforms
        self.samples = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            image_path = os.path.join(folder_path, 'image.tif')
            mask_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.startswith('class') and f.endswith('.tif')
            ]

            if os.path.exists(image_path) and mask_paths:
                self.samples.append((image_path, mask_paths))

        print(f"Found {len(self.samples)} samples in {root_dir}")

    def __len__(self):
        """Returns dataset size."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetches sample at index idx."""
        image_path, mask_paths = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}")
            return None, None

        boxes = []
        labels = []
        masks = []

        # Process each mask file associated with the image
        for mask_path in mask_paths:
            match = re.search(r'class(\d+)\.tif', os.path.basename(mask_path))
            if not match:
                print(
                    f"Warning: Mask file {mask_path} does not follow expected"
                    " naming convention."
                )
                continue

            try:
                label = int(match.group(1))
                image_mask = read_maskfile(mask_path)
                if image_mask is None:
                    continue

                if image_mask.ndim == 3:
                    image_mask = image_mask.squeeze(0)

                # Find unique object IDs in the mask (excluding background 0)
                object_ids = np.unique(image_mask)
                object_ids = object_ids[object_ids != 0]

                # Process each object instance in the mask
                for object_id in object_ids:
                    # Create a binary mask for the current object instance
                    cur_mask = (image_mask == object_id).astype(np.uint8)

                    # Find bounding box coordinates
                    pos = np.where(cur_mask)
                    if len(pos[0]) == 0 or len(pos[1]) == 0:
                        continue

                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    if xmax <= xmin or ymax <= ymin:
                        print(
                            f"Warning: Invalid bounding box for object"
                            f" {object_id} in mask {mask_path}"
                        )
                        continue

                    box = [xmin, ymin, xmax, ymax]
                    boxes.append(box)
                    labels.append(label)
                    masks.append(cur_mask)

            except Exception as e:
                print(f"Error processing mask file {mask_path}: {e}")
                continue

        image_id_tensor = torch.tensor([idx])

        if self.transforms:
            transform_input = {
                'image': image_np,
                'bboxes': boxes,
                'category_ids': labels,
                'masks': masks
            }

            try:
                transformed = self.transforms(**transform_input)
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['category_ids']
                masks = transformed['masks']
            except Exception as e:
                print(f"Error applying transforms to sample {idx}: {e}")
                return None, None

        boxes_tensor = (
            torch.as_tensor(boxes, dtype=torch.float32) if boxes else
            torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.as_tensor(labels, dtype=torch.int64) if labels else
            torch.zeros((0,), dtype=torch.int64)
        )

        masks_tensor_list = [
            torch.as_tensor(mask, dtype=torch.uint8) for mask in masks
        ]
        transformed_height, transformed_width = image.shape[1], image.shape[2]
        masks_tensor = (
            torch.stack(masks_tensor_list) if masks_tensor_list else
            torch.zeros(
                (0, transformed_height, transformed_width), dtype=torch.uint8
            )
        )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": image_id_tensor,
            "area": (boxes_tensor[:, 3] - boxes_tensor[:, 1]) *
                    (boxes_tensor[:, 2] - boxes_tensor[:, 0]),
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        }

        return image, target


def create_gt_coco(dataset_subset):
    """Creates COCO ground truth dict from dataset subset."""
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "class1"},
            {"id": 2, "name": "class2"},
            {"id": 3, "name": "class3"},
            {"id": 4, "name": "class4"},
        ]
    }
    annotation_id = 1

    for idx in range(len(dataset_subset)):
        image, target = dataset_subset[idx]

        if image is None or target is None:
            continue

        image_id = target["image_id"].item()
        height, width = image.shape[1], image.shape[2]

        coco_dict["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"image_{image_id}.tif",
        })

        # Add annotations for each instance in the target
        for i in range(len(target["boxes"])):
            box = target["boxes"][i].tolist()
            xmin, ymin, xmax, ymax = box

            bbox_coco = [xmin, ymin, xmax - xmin, ymax - ymin]
            category_id = target["labels"][i].item()
            mask = target["masks"][i].cpu().numpy()

            binary_mask = (mask > 0).astype(np.uint8)
            area = int(np.sum(binary_mask))

            if area == 0:
                continue

            rle = encode_mask(binary_mask)

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                "segmentation": rle,
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1

    print(f"Created COCO dict with {len(coco_dict['images'])} images and"
          f" {len(coco_dict['annotations'])} annotations.")
    return coco_dict

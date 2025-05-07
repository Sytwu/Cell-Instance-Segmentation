# utils.py
import os
import random
import numpy as np
import torch
import skimage.io as sio
from pycocotools import mask as mask_utils


def set_seed(seed):
    """Sets seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def decode_maskobj(mask_obj):
    """Decodes a COCO RLE (Run-Length Encoding) mask object."""
    return mask_utils.decode(mask_obj)


def read_maskfile(filepath):
    """Reads a mask file (image) and returns it as a NumPy array."""
    mask_array = sio.imread(filepath)
    return mask_array


def encode_mask(binary_mask):
    """Encodes a binary mask (NumPy array) to COCO RLE format."""
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def collate_fn(batch):
    return tuple(zip(*batch))

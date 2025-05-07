# config.py
import argparse
import os
import json

def get_config():
    """Parses args, sets up output dir."""
    parser = argparse.ArgumentParser(description="Mask R-CNN Config")

    # Data paths
    parser.add_argument('--root_dir', type=str, default='hw3-data-release/train',
                        help='Root dir for training data')
    parser.add_argument('--test_dir', type=str, default='hw3-data-release/test_release',
                        help='Dir containing test images')
    parser.add_argument('--test_json', type=str, default='hw3-data-release/test_image_name_to_ids.json',
                        help='JSON mapping test image names to IDs')

    # Model parameters
    parser.add_argument('--model_arch', type=str, default='MaskRCNN',
                        choices=['MaskRCNN', 'MaskRCNN_v2', 'MaskRCNN_ConvNeXt', 'MaskRCNN_Cell'],
                        help='Model architecture to use')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes including background')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights for the model')
    # Add model specific parameters
    parser.add_argument('--rpn_post_nms_train', type=int, default=2000,
                        help='RPN post-NMS top N proposals during training')
    parser.add_argument('--rpn_post_nms_test', type=int, default=1000,
                        help='RPN post-NMS top N proposals during testing')
    parser.add_argument('--box_detections_per_img', type=int, default=100,
                        help='Maximum number of detections per image')


    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='Base learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--valid_split', type=float, default=0.2,
                        help='Fraction for validation split')

    # LR Scheduler parameters
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler type (step, cosine, none)')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for StepLR scheduler')
    # For Cosine Annealing, T_max is usually num_epochs
    # parser.add_argument('--lr_T_max', type=int, default=None,
    #                     help='T_max for CosineAnnealingLR scheduler (defaults to num_epochs)')


    # Output directory
    parser.add_argument('--out_dir_base', type=str, default='results',
                        help='Base dir for saving results')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g., "cuda", "cpu"). Auto-detects if None.')

    args = parser.parse_args()

    # Output Directory Handling
    i = 1
    while True:
        args.out_dir = os.path.join(args.out_dir_base, f'version{i}')
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            print(f"Created output directory: {args.out_dir}")
            break
        i += 1

    # Save Config
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved configuration to {config_path}")

    return args


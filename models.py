import torch
import torch.nn as nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN as TorchvisionMaskRCNN,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import (
    BackboneWithFPN,
    LastLevelMaxPool,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class MaskRCNN(nn.Module):
    """Mask R-CNN model wrapper (ResNet50-FPN)."""
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        """Initializes Mask R-CNN with configurable parameters."""
        super().__init__()

        # Load pre-trained or new model
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            box_detections_per_img=box_detections_per_img,
        )

        # Modify Box Predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # num_classes includes background, so it's actual classes + 1
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                               num_classes)

        # Modify Mask Predictor
        in_features_mask = \
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    def get_parameter_size(self):
        """Prints model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """Configures AdamW optimizer."""
        low_lr = base_lr * 0.1
        params = [
            {'params': self.model.backbone.parameters(), 'lr': low_lr},
            {'params': self.model.rpn.parameters(), 'lr': base_lr},
            {'params': self.model.roi_heads.parameters(), 'lr': base_lr}
        ]
        optimizer = torch.optim.AdamW(params, lr=base_lr,
                                      weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """Forward pass."""
        return self.model(images, targets)


class MaskRCNN_v2(MaskRCNN):
    """Mask R-CNN model wrapper (ResNet50-FPN v2)."""
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        super(MaskRCNN, self).__init__()

        # Load pre-trained or new v2 model
        weights = None
        if pretrained:
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            box_detections_per_img=box_detections_per_img,
        )

        # Modify Box Predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                               num_classes)

        # Modify Mask Predictor
        in_features_mask = \
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )


# Define a customized ConvNeXt model to expose intermediate features for FPN
class ModifiedConvNeXt(nn.Module):
    """ConvNeXt backbone wrapper to expose intermediate features."""
    def __init__(self, original_model):
        super().__init__()
        # Output C1
        self.stage1 = nn.Sequential(original_model.features[0],
                                    original_model.features[1])
        # Output C2
        self.stage2 = nn.Sequential(original_model.features[2],
                                    original_model.features[3])
        # Output C3
        self.stage3 = nn.Sequential(original_model.features[4],
                                    original_model.features[5])
        # Output C4
        self.stage4 = nn.Sequential(original_model.features[6],
                                    original_model.features[7])

        # Define the output channels for each stage
        self._out_channels = [128, 256, 512, 1024]

    def forward(self, x):
        """Extract intermediate feature maps for FPN."""
        out1 = self.stage1(x)  # C1
        out2 = self.stage2(out1)  # C2
        out3 = self.stage3(out2)  # C3
        out4 = self.stage4(out3)  # C4

        return {
            'stage1': out1,
            'stage2': out2,
            'stage3': out3,
            'stage4': out4
        }

    def out_channels(self):
        """Returns the output channels of the stages."""
        return self._out_channels


# Define Mask R-CNN model using ConvNeXt as the backbone
class MaskRCNN_ConvNeXt(nn.Module):
    """Mask R-CNN model with ConvNeXt backbone and FPN."""
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        super().__init__()

        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        original_convnext = convnext_base(weights=weights)
        modified_convnext = ModifiedConvNeXt(original_convnext)

        return_layers = {
            'stage1': '0',
            'stage2': '1',
            'stage3': '2',
            'stage4': '3'
        }

        backbone_with_fpn = BackboneWithFPN(
            modified_convnext,
            in_channels_list=modified_convnext.out_channels(),
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
            return_layers=return_layers
        )

        anchor_sizes = tuple((x, ) for x in [32, 64, 128, 256, 512])
        aspect_ratios = tuple(((0.5, 1.0, 2.0),) * len(anchor_sizes))
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['1', '2', '3', '4'],
            output_size=7,
            sampling_ratio=2
        )

        # Define Mask ROI Pooler
        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=['1', '2', '3', '4'],
            output_size=14,
            sampling_ratio=2
        )

        self.model = TorchvisionMaskRCNN(
            backbone=backbone_with_fpn,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            detections_per_img=box_detections_per_img
        )

    def get_parameter_size(self):
        """Prints model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """Configures AdamW optimizer with differential LR."""
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        params = [
            {'params': backbone_params, 'lr': base_lr},
            {'params': other_params, 'lr': base_lr}
        ]

        optimizer = torch.optim.AdamW(params, lr=base_lr,
                                      weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """Forward pass."""
        return self.model(images, targets)


# Define a modified Mask R-CNN for Cell Segmentation
class MaskRCNN_Cell(nn.Module):
    """Mask R-CNN model with modifications for cell instance segmentation."""
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        """Initializes Mask R-CNN Cell variant."""
        super().__init__()
        weights = None
        if pretrained:
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

        standard_model_for_backbone = maskrcnn_resnet50_fpn(
            weights=weights
        )
        backbone = standard_model_for_backbone.backbone

        self.model = TorchvisionMaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            detections_per_img=box_detections_per_img
        )

        in_channels_mask = \
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels

        new_hidden_channels = 512

        original_mask_layers = list(
            self.model.roi_heads.mask_predictor.children()
        )

        new_mask_layers = []
        current_in_channels = in_channels_mask

        for i, layer in enumerate(original_mask_layers):
            if isinstance(layer, nn.Conv2d):
                new_out_channels = new_hidden_channels
                new_conv = nn.Conv2d(current_in_channels, new_out_channels,
                                     kernel_size=layer.kernel_size,
                                     stride=layer.stride,
                                     padding=layer.padding)
                new_mask_layers.append(new_conv)
                current_in_channels = new_out_channels

            elif isinstance(layer, nn.ReLU):
                new_mask_layers.append(layer)

            elif isinstance(layer, nn.ConvTranspose2d):
                if i == len(original_mask_layers) - 1:
                    extra_conv = nn.Conv2d(current_in_channels,
                                           new_hidden_channels,
                                           kernel_size=3, stride=1, padding=1)
                    new_mask_layers.append(extra_conv)
                    new_mask_layers.append(nn.ReLU(inplace=True))
                    current_in_channels = new_hidden_channels

                    final_conv_transpose = nn.ConvTranspose2d(
                        current_in_channels, num_classes,
                        kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding
                    )
                    new_mask_layers.append(final_conv_transpose)
                else:
                    print(
                        f"Warning: Unexpected ConvTranspose2d layer at index"
                        f" {i}"
                    )
                    new_mask_layers.append(layer)
                    current_in_channels = layer.out_channels

        self.model.roi_heads.mask_predictor = nn.Sequential(*new_mask_layers)

    def get_parameter_size(self):
        """Prints model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """Configures AdamW optimizer with differential LR."""
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        params = [
            {'params': backbone_params, 'lr': base_lr * 0.1},
            {'params': other_params, 'lr': base_lr}
        ]

        optimizer = torch.optim.AdamW(params, lr=base_lr,
                                      weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """Forward pass."""
        return self.model(images, targets)

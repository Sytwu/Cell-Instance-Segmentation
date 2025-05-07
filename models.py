# models.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN,
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
    """
    Mask R-CNN model wrapper using ResNet50-FPN backbone.
    This class provides a standard Mask R-CNN model with options to
    configure pre-trained weights and NMS/detection parameters.
    """
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        """
        Initializes the Mask R-CNN model.

        Args:
            num_classes (int): Number of output classes (including background).
            pretrained (bool): If True, loads weights pre-trained on COCO.
            rpn_post_nms_train (int): Max number of RPN proposals after NMS in training.
            rpn_post_nms_test (int): Max number of RPN proposals after NMS in testing.
            box_detections_per_img (int): Max number of final detections per image.
        """
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            box_detections_per_img=box_detections_per_img,
        )

        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        # Replace the pre-trained mask predictor with a new one.
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer_mask = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer_mask, num_classes
        )

    def get_parameter_size(self):
        """Prints the total and trainable parameter counts of the model."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """
        Configures and returns an AdamW optimizer with differential learning rates
        for backbone and other parts of the model.
        """
        low_lr = base_lr * 0.1
        params = [
            {'params': self.model.backbone.parameters(), 'lr': low_lr},
            {'params': self.model.rpn.parameters(), 'lr': base_lr},
            {'params': self.model.roi_heads.parameters(), 'lr': base_lr}
        ]
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """
        Defines the forward pass of the model.
        During training, it expects images and targets and returns a dict of losses.
        During inference, it expects only images and returns a list of detections.
        """
        return self.model(images, targets)


class MaskRCNN_v2(MaskRCNN):
    """
    Mask R-CNN model wrapper using ResNet50-FPN v2 backbone.
    This version utilizes the improved v2 architecture of ResNet50-FPN.
    Inherits optimizer, parameter counting, and forward pass from the base MaskRCNN class.
    """
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        """
        Initializes the Mask R-CNN v2 model.
        Args are the same as the base MaskRCNN class.
        """
        super(MaskRCNN, self).__init__()

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            box_detections_per_img=box_detections_per_img,
        )

        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer_mask = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer_mask, num_classes
        )


class ModifiedConvNeXt(nn.Module):
    """
    A modified ConvNeXt backbone to expose intermediate feature maps for FPN.
    ConvNeXt, by default, doesn't return intermediate features in a way
    directly usable by torchvision's FPN. This class wraps it.
    """
    def __init__(self, original_model):
        super().__init__()
        self.stage1 = nn.Sequential(original_model.features[0], original_model.features[1])
        self.stage2 = nn.Sequential(original_model.features[2], original_model.features[3])
        self.stage3 = nn.Sequential(original_model.features[4], original_model.features[5])
        self.stage4 = nn.Sequential(original_model.features[6], original_model.features[7])

        self._out_channels = [128, 256, 512, 1024]  # C1, C2, C3, C4 channels

    def forward(self, x):
        """
        Extracts intermediate feature maps from the ConvNeXt stages.
        Returns a dictionary mapping stage names to their output tensors,
        which is the format expected by BackboneWithFPN.
        """
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {'stage1': s1, 'stage2': s2, 'stage3': s3, 'stage4': s4}

    def out_channels(self):
        """Returns the list of output channel numbers for each stage."""
        return self._out_channels


class MaskRCNN_ConvNeXt(nn.Module):
    """
    Mask R-CNN model with a ConvNeXt-Base backbone and FPN.
    This model demonstrates how to integrate a custom backbone (ConvNeXt)
    into the Mask R-CNN framework.
    """
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        super().__init__()

        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        original_convnext = convnext_base(weights=weights)
        modified_convnext = ModifiedConvNeXt(original_convnext)

        return_layers = {'stage1': '0', 'stage2': '1', 'stage3': '2', 'stage4': '3'}

        # Create the FPN-enhanced backbone.
        backbone_with_fpn = BackboneWithFPN(
            modified_convnext,
            return_layers=return_layers,
            in_channels_list=modified_convnext.out_channels(),
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )

        # Define the RPN anchor generator.
        anchor_sizes = tuple((x,) for x in [32, 64, 128, 256, 512])
        aspect_ratios = tuple(((0.5, 1.0, 2.0),) * len(anchor_sizes))
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        # Define RoI Aligners for box and mask heads.
        roi_pooler_featmap_names = ['0', '1', '2', '3']
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=roi_pooler_featmap_names,
            output_size=7,
            sampling_ratio=2
        )
        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=roi_pooler_featmap_names,
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
        """Prints the total and trainable parameter counts of the model."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """
        Configures AdamW optimizer with differential LR for ConvNeXt backbone.
        ConvNeXt backbone parameters might benefit from a different LR.
        """
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        # A common strategy is to use a lower LR for the backbone.
        params = [
            {'params': backbone_params, 'lr': base_lr * 0.1},
            {'params': other_params, 'lr': base_lr}
        ]
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """Defines the forward pass of the model."""
        return self.model(images, targets)


class MaskRCNN_Cell(nn.Module):
    """
    Mask R-CNN model specifically adapted for cell instance segmentation.
    This variant uses a standard ResNet50-FPN backbone but modifies
    the mask head for potentially better performance on cell-like structures.
    """
    def __init__(self, num_classes=5, pretrained=True,
                 rpn_post_nms_train=2000, rpn_post_nms_test=1000,
                 box_detections_per_img=100):
        super().__init__()

        temp_model_for_backbone = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        )
        backbone = temp_model_for_backbone.backbone

        self.model = TorchvisionMaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_post_nms_top_n_train=rpn_post_nms_train,
            rpn_post_nms_top_n_test=rpn_post_nms_test,
            detections_per_img=box_detections_per_img
        )

        in_channels_mask_head = backbone.out_channels
        new_hidden_channels = 512

        layers = []
        current_channels = in_channels_mask_head
        for _ in range(4):
            layers.append(nn.Conv2d(current_channels, new_hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = new_hidden_channels

        layers.append(nn.Conv2d(current_channels, new_hidden_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        current_channels = new_hidden_channels

        layers.append(nn.ConvTranspose2d(current_channels, num_classes, kernel_size=2, stride=2, padding=0))

        self.model.roi_heads.mask_predictor = nn.Sequential(*layers)

    def get_parameter_size(self):
        """Prints the total and trainable parameter counts of the model."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=1e-3, weight_decay=1e-4):
        """
        Configures AdamW optimizer with differential LR.
        Backbone parameters often benefit from a lower learning rate.
        """
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
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        """Defines the forward pass of the model."""
        return self.model(images, targets)

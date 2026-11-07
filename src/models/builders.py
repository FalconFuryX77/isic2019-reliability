"""
Model builders — ResNet-50 and EfficientNet-B3 with a configurable head.

Head architecture (as specified in the project plan):
    pretrained backbone
        -> Global Average Pooling
        -> BatchNorm
        -> Dense (relu)
        -> Dropout
        -> num_classes logits

Outputs RAW LOGITS (no softmax) so temperature scaling can be applied later.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class _ClassifierHead(nn.Module):
    """GAP -> BN -> Dense(relu) -> Dropout -> logits."""

    def __init__(self, in_features, num_classes, dense_size=128, dropout=0.5, use_batchnorm=True):
        super().__init__()
        layers = []
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(in_features))
        layers.append(nn.Linear(in_features, dense_size))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dense_size, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class SkinLesionModel(nn.Module):
    """Wrapper that adapts a torchvision backbone into the project head."""

    def __init__(self, backbone_name="resnet50", num_classes=8, pretrained=True,
                 dense_size=128, dropout=0.5, use_batchnorm=True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = tv_models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-2])  # keep conv blocks
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_features = base.fc.in_features
        elif backbone_name == "efficientnet_b3":
            weights = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base = tv_models.efficientnet_b3(weights=weights)
            # Use all feature blocks; the classifier is replaced by our head.
            self.features = base.features
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            in_features = base.classifier[1].in_features  # 1536 for B3
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.head = _ClassifierHead(
            in_features, num_classes, dense_size=dense_size,
            dropout=dropout, use_batchnorm=use_batchnorm)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        if self.backbone_name == "resnet50":
            x = torch.flatten(x, 1)
        return self.head(x)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_fraction(self, fraction):
        """Unfreeze the last `fraction` of backbone modules (by parameter index)."""
        params = list(self.features.parameters())
        n = len(params)
        freeze_until = int(n * (1 - fraction))
        for i, p in enumerate(params):
            p.requires_grad = i >= freeze_until

    def enable_dropout(self, enabled=True):
        """Toggle dropout modules for MC Dropout inference."""
        def _apply(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train() if enabled else module.eval()
        self.apply(_apply)


def build_model(backbone_name="resnet50", num_classes=8, pretrained=True,
                dense_size=128, dropout=0.5, use_batchnorm=True, device=None):
    model = SkinLesionModel(
        backbone_name=backbone_name, num_classes=num_classes,
        pretrained=pretrained, dense_size=dense_size,
        dropout=dropout, use_batchnorm=use_batchnorm)
    if device is not None:
        model = model.to(device)
    return model


def get_gradcam_target_layer(model: SkinLesionModel):
    """Return the last conv layer of the backbone for Grad-CAM."""
    if model.backbone_name == "resnet50":
        # last conv block, final conv
        return model.features[-1][-1].conv3
    elif model.backbone_name == "efficientnet_b3":
        return model.features[-1][0]
    raise ValueError(model.backbone_name)
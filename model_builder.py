"""
model_builder.py
Creates the AisleEye model using transfer learning (EfficientNet-B0).
"""

import torch
from torch import nn
import torchvision

def create_aisleye_model(num_classes: int = 3):
    """
    Creates an EfficientNet-B0 model with a custom classifier head
    for the 3 tray-table classes.

    Args:
        num_classes: Number of output classes (default 3).

    Returns:
        model: The modified EfficientNet-B0 model.
        weights: The pre-trained weights object (useful for transforms).
    """
    # 1. Load pre-trained EfficientNet-B0 weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    # 2. Freeze the base feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Replace the classifier head for our 3 classes
    num_features = model.classifier[1].in_features  # usually 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=num_features, out_features=num_classes)
    )

    return model, weights
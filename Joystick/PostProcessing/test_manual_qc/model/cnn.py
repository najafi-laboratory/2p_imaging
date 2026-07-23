"""
model/cnn.py

Binary classifier for ROI QC using a pretrained lightweight backbone.

Input:
    (B, 2, H, W)
    - channel 0: normalized mean image
    - channel 1: ROI binary mask

Output:
    p(good ROI)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ROICNN(nn.Module):

    def __init__(self, pretrained=True, in_channels=2, freeze_backbone=True):
        super().__init__()
        self.in_channels = in_channels
        self.freeze_backbone = freeze_backbone

        if pretrained:
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )

        else:
            self.backbone = models.resnet18(
                weights=None
            )

        self._adapt_first_conv(in_channels)
        self._replace_classifier_head()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.conv1.weight.requires_grad = True
            self.backbone.fc.weight.requires_grad = True
            self.backbone.fc.bias.requires_grad = True

    def _adapt_first_conv(self, in_channels):
        """
        Modify ResNet first convolution:
        RGB input (3 channels) -> 2-channel input
        channel 0: mean image
        channel 1: ROI mask
        """

        if in_channels == 3:
            return

        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # initialize from ImageNet RGB weights
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                # average RGB filters
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)

                # duplicate for new channels
                new_conv.weight.copy_(
                    mean_weight.repeat(1, in_channels, 1, 1)
                )
            else:
                nn.init.kaiming_normal_(
                    new_conv.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

        self.backbone.conv1 = new_conv

    def _replace_classifier_head(self):
        """
        Replace ImageNet 1000-class classifier
        with binary classifier.
        """
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            1
        )

    def forward(self, x):
        """
        x: (B, 2, H, W)
        returns:
            logits: (B,1)
        """
        return self.backbone(x)


class ROIModel:

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = ROICNN(pretrained=False).to(device)
        self.sigmoid = nn.Sigmoid()

    def predict_proba(self, x):
        """
        x: torch tensor (B, 2, H, W)
        returns: probability of good ROI
        """
        self.model.eval()

        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            # print("logit:", logits[:5])
            probs = self.sigmoid(logits)

        return probs.cpu().numpy()

    def load(self, path):
        result = self.model.load_state_dict(torch.load(path, map_location=self.device), strict=True)
        print("Checkpoint loaded:")
        print(result)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
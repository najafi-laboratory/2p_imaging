"""
model/cnn.py

Multimodal ROI QC classifier.

Inputs:
    image:
        (B, 2, H, W)

    signal:
        (B, 2, T)
        channel 0: ROI fluorescence signal
        channel 1: neuropil signal

    metrics:
        (B, M)
        morphology / fluorescence QC metrics

Output:
    logits: (B, 1)

Supports modality ablations:

    image only
    signal only
    metrics only
    image + signal
    image + metrics
    signal + metrics
    image + signal + metrics
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# IMAGE ENCODER
# ============================================================

class ImageEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True,
        in_channels=2,
        freeze_backbone=True,
        embedding_dim=256,
    ):
        super().__init__()

        if pretrained:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            backbone = models.resnet18(
                weights=None
            )

        self._adapt_first_conv(
            backbone,
            in_channels
        )

        resnet_features = backbone.fc.in_features

        # Remove ImageNet classifier
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(
                resnet_features,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        if freeze_backbone:

            for param in self.backbone.parameters():
                param.requires_grad = False

            # Allow adapted input convolution to learn
            self.backbone.conv1.weight.requires_grad = True

    def _adapt_first_conv(
        self,
        backbone,
        in_channels,
    ):

        if in_channels == 3:
            return

        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():

            if old_conv.weight.shape[1] == 3:

                mean_weight = old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )

                new_conv.weight.copy_(
                    mean_weight.repeat(
                        1,
                        in_channels,
                        1,
                        1
                    )
                )

            else:

                nn.init.kaiming_normal_(
                    new_conv.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

        backbone.conv1 = new_conv

    def forward(self, x):

        x = self.backbone(x)

        x = self.projection(x)

        return x


# ============================================================
# SIGNAL ENCODER
# ============================================================

class SignalEncoder(nn.Module):

    def __init__(
        self,
        in_channels=2,
        embedding_dim=128,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),
        )

    def forward(self, x):

        # Expected:
        # (B, 2, T)

        if x.dim() != 3:
            raise ValueError(
                f"Expected signal shape (B, C, T), got {x.shape}"
            )

        x = self.features(x)

        x = self.projection(x)

        return x


# ============================================================
# METRICS ENCODER
# ============================================================

class MetricsEncoder(nn.Module):

    def __init__(
        self,
        num_metrics,
        embedding_dim=64,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                num_metrics,
                128
            ),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2),
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# MULTIMODAL ROI QC MODEL
# ============================================================

class ROICNN(nn.Module):

    def __init__(
        self,
        num_metrics,
        pretrained=True,
        in_channels=2,
        freeze_backbone=True,

        # -----------------------------------------------------
        # Ablation switches
        # -----------------------------------------------------

        use_image=True,
        use_signal=True,
        use_metrics=True,
    ):
        super().__init__()

        self.use_image = use_image
        self.use_signal = use_signal
        self.use_metrics = use_metrics

        if not any([
            use_image,
            use_signal,
            use_metrics,
        ]):
            raise ValueError(
                "At least one modality must be enabled."
            )

        # -----------------------------------------------------
        # Image branch
        # -----------------------------------------------------

        if self.use_image:

            self.image_encoder = ImageEncoder(
                pretrained=pretrained,
                in_channels=in_channels,
                freeze_backbone=freeze_backbone,
                embedding_dim=256,
            )

        # -----------------------------------------------------
        # Signal branch
        # -----------------------------------------------------

        if self.use_signal:

            self.signal_encoder = SignalEncoder(
                in_channels=2,
                embedding_dim=128,
            )

        # -----------------------------------------------------
        # Metrics branch
        # -----------------------------------------------------

        if self.use_metrics:

            self.metrics_encoder = MetricsEncoder(
                num_metrics=num_metrics,
                embedding_dim=64,
            )

        # -----------------------------------------------------
        # Determine fusion dimension
        # -----------------------------------------------------

        fusion_dim = 0

        if self.use_image:
            fusion_dim += 256

        if self.use_signal:
            fusion_dim += 128

        if self.use_metrics:
            fusion_dim += 64

        # -----------------------------------------------------
        # Fusion classifier
        # -----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        image=None,
        signal=None,
        metrics=None,
    ):

        features = []

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        if self.use_image:

            if image is None:
                raise ValueError(
                    "Image modality enabled but image was None."
                )

            image_features = self.image_encoder(
                image
            )

            features.append(image_features)

        # -----------------------------------------------------
        # Signal
        # -----------------------------------------------------

        if self.use_signal:

            if signal is None:
                raise ValueError(
                    "Signal modality enabled but signal was None."
                )

            signal_features = self.signal_encoder(
                signal
            )

            features.append(signal_features)

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        if self.use_metrics:

            if metrics is None:
                raise ValueError(
                    "Metrics modality enabled but metrics was None."
                )

            metric_features = self.metrics_encoder(
                metrics
            )

            features.append(metric_features)

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        if len(features) == 1:

            fused = features[0]

        else:

            fused = torch.cat(
                features,
                dim=1
            )

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# MODEL WRAPPER
# ============================================================

class ROIModel:

    def __init__(
        self,
        num_metrics,
        device=None,
        use_image=True,
        use_signal=True,
        use_metrics=True,
    ):

        if device is None:

            if torch.cuda.is_available():
                device = "cuda"

            elif torch.backends.mps.is_available():
                device = "mps"

            else:
                device = "cpu"

        self.device = device

        self.model = ROICNN(
            num_metrics=num_metrics,
            pretrained=False,
            use_image=use_image,
            use_signal=use_signal,
            use_metrics=use_metrics,
        ).to(self.device)

        self.sigmoid = nn.Sigmoid()

    def predict_proba(
        self,
        image=None,
        signal=None,
        metrics=None,
    ):

        self.model.eval()

        with torch.no_grad():

            if image is not None:
                image = image.to(self.device)

            if signal is not None:
                signal = signal.to(self.device)

            if metrics is not None:
                metrics = metrics.to(self.device)

            logits = self.model(
                image=image,
                signal=signal,
                metrics=metrics,
            )

            probs = self.sigmoid(
                logits
            )

        return probs.cpu().numpy()

    def load(self, path):

        # result = self.model.load_state_dict(
        #     torch.load(
        #         path,
        #         map_location=self.device,
        #         weights_only=False,
        #     ),
        #     strict=True
        # )

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        
        result = self.model.load_state_dict(
            checkpoint["model_state"],
            strict=True
        )

        print("Checkpoint loaded:")
        print(result)
        return checkpoint

    def save(self, path):

        torch.save(
            self.model.state_dict(),
            path
        )
import torch
import torch.nn as nn
import timm


class CorruptionAwareSwin(nn.Module):
    def __init__(
        self,
        num_classes=43,
        num_corruptions=15,  # 14 corruptions + clean
        num_severities=5,
        pretrained=False,
    ):
        super().__init__()

        # 1. Swin-Tiny feature extractor
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = 768

        # 2. Predict the corruption condition
        self.corruption_head = nn.Linear(
            feature_dim, num_corruptions
        )

        # 3. Predict severity
        self.severity_head = nn.Linear(
            feature_dim, num_severities
        )

        # 4. Feature-routing module
        condition_dim = num_corruptions + num_severities

        self.router = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.GELU(),
            nn.Linear(128, feature_dim * 2),
        )

        # Start with zero feature adjustment.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # 5. Final traffic-sign classifier
        self.classifier = nn.Linear(
            feature_dim, num_classes
        )

    def forward(self, images):
        # Extract Swin features
        features = self.backbone(images)

        # Predict image condition
        corruption_logits = self.corruption_head(features)
        severity_logits = self.severity_head(features)

        # Use predictions, not ground-truth labels, for routing.
        corruption_probs = torch.softmax(
            corruption_logits, dim=1
        )
        severity_probs = torch.softmax(
            severity_logits, dim=1
        )

        condition = torch.cat(
            [corruption_probs, severity_probs], dim=1
        )

        # Generate feature scaling and shifting parameters
        gamma_beta = self.router(condition)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Adjust the features
        adjusted_features = (
            features * (1.0 + gamma) + beta
        )

        # Predict the traffic-sign class
        class_logits = self.classifier(adjusted_features)

        return {
            "class_logits": class_logits,
            "corruption_logits": corruption_logits,
            "severity_logits": severity_logits,
        }
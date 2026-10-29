
import torch
import torch.nn as nn
import timm


class SwinBaseline(nn.Module):
    def __init__(self, num_classes=43, pretrained=True):
        super().__init__()

        # Swin-Tiny backbone returns a 768-dimensional feature.
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )

        # Final feature vector -> traffic-sign class scores.
        self.classifier = nn.Linear(
            in_features=768,
            out_features=num_classes,
        )

    def forward(self, images):
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SwinBaseline().to(device)
    model.eval()

    sample_images = torch.randn(
        4, 3, 224, 224, device=device
    )

    with torch.no_grad():
        logits = model(sample_images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    print("Device:", device)
    print("Input shape:", sample_images.shape)
    print("Output shape:", logits.shape)
    print("Predicted classes:", predictions.tolist())
    print("Probabilities sum:", probabilities[0].sum().item())

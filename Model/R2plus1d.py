from torchvision.models.video import r2plus1d_18
import torch.nn as nn


class R2plus1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = r2plus1d_18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out

import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig


class ViViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ViT_spatial = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.temporal_Encoder = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=768, nhead=8, dim_feedforward=2048, activation='gelu', batch_first=True),
            nn.TransformerEncoderLayer(
                d_model=768, nhead=8, dim_feedforward=2048, activation='gelu', batch_first=True),
            nn.TransformerEncoderLayer(
                d_model=768, nhead=8, dim_feedforward=2048, activation='gelu', batch_first=True))
        #self.temporal_token = nn.Parameter(torch.randn(1, 1, 192))
        self.fc1 = nn.Linear(768, 192)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(192, 6)

    def forward(self, x):
        ViT_out = torch.zeros((x.shape[0], x.shape[1], 768)).cuda()
        temporal_token = nn.Parameter(torch.randn(x.shape[0], 1, 768)).cuda()
        for i in range(x.shape[1]):
            out = self.ViT_spatial(x[:, i, :, :, :])
            ViT_out[:, i, :] = out.pooler_output
        temporal_encoder_input = torch.cat((ViT_out, temporal_token), dim=1)
        out = self.temporal_Encoder(temporal_encoder_input)
        out = out[:, 0, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    model = ViViT()
    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name)
    #input = torch.zeros((4, 8, 3, 224, 224))
    #output = model(input)
    # print(output)
    # print(model)

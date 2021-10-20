import torch
import torch.nn as nn
from TimeSformer.timesformer.models.vit import TimeSformer
#model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
#                    pretrained_model=r'C:\NTHU\MISLAB_work\Facial_Expression_Recognition\TimeSformer\TimeSformer_divST_8x32_224_K400.pyth')
#print(model)


class TimeSformer_FER(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                                      pretrained_model=r'C:\NTHU\MISLAB_work\Facial_Expression_Recognition\TimeSformer\TimeSformer_divST_8x32_224_K400.pyth')
        self.pretrained.model.blocks.require_grad = False
        self.pretrained.model.head = nn.Linear(768, 6)

    def forward(self, x):
        out = self.pretrained(x)
        return out


if __name__ == "__main__":
    input = torch.zeros((1, 3, 8, 224, 224))
    model = TimeSformer_FER()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #output = model(input)
    # print(output)

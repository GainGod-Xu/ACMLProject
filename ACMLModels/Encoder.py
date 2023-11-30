import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, model, trainable):
        super().__init__()

        self.model = model

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input):
        return self.model(input)

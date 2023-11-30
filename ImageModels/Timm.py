import torch
import timm
from DatasetModels.ImageDataset import ImageDataset
from torch import nn
from Utils.CMRPConfig import CMRPConfig
import pandas as pd
import pandas as pd
from torch.utils.data import DataLoader
class TimmModel(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name='resnet50',
                 pretrained=CMRPConfig.mr1_model_pretrained,
                 trainable=CMRPConfig.mr1_model_trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)



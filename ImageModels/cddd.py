r""" Modified from the paper `"Img2Mol – accurate SMILES recognition from molecular graphical depictions"
    <https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc01839f>`_ paper
    
    Original implementation: https://github.com/bayer-science-for-a-better-life/Img2Mol/tree/main.
    """


import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union, List, Optional
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from rdkit import Chem

MODEL_CONFIGS: List = [[128, 7, 3, 4],
                        [256, 5, 1, 1],
                        [384, 5, 1, 1],
                        'M',
                        [384, 3, 1, 1],
                        [384, 3, 1, 1],
                        'M',
                        [512, 3, 1, 1],
                        [512, 3, 1, 1],
                        [512, 3, 1, 1],
                        'M']


def make_layers(cfg: Optional[List[Union[str, int]]] = None,
                batch_norm: bool = False) -> nn.Sequential:
    """
    Helper function to create the convolutional layers for the Img2Mol model to be passed into a nn.Sequential module.
    :param cfg: list populated with either a str or a list, where the str object refers to the pooling method and the
                list object will be unrolled to obtain the convolutional-filter parameters.
                Defaults to the `MODEL_CONFIGS` list.
    :param batch_norm: boolean of batch normalization should be used in-between conv2d and relu activation.
                       Defaults to False
    :return: torch.nn.Sequential module as feature-extractor
    """
    if cfg is None:
        cfg = MODEL_CONFIGS

    layers: List[nn.Module] = []

    in_channels = 1
    for v in cfg:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                units, kern_size, stride, padding = v
                conv2d = nn.Conv2d(in_channels, units, kernel_size=kern_size, stride=stride, padding=padding)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(units), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = units

    model = nn.Sequential(*layers)
    return model


class Img2Mol(pl.LightningModule):
    """
    Wraps the Img2Mol model into pytorch lightning for easy training and inference
    """
    def __init__(self, learning_rate: float = 1e-4,
                 batch_norm: bool = False,
                 trainable: bool = False):
        super().__init__()
        self.learning_rate = learning_rate

        # convolutional NN for feature extraction
        self.features = make_layers(cfg=MODEL_CONFIGS, batch_norm=batch_norm)
        # fully-connected network for classification based on CNN feature extractor
        self.classifier = nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 512),
            nn.Tanh(),
        )

        self._initialize_weights()
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def training_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, cddd = batch
        cddd_hat = self(x)
        loss = F.mse_loss(cddd_hat, cddd)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# class Img2MolInference(object):
#     """
#     Inference Class
#     """
#     def __init__(
#         self,
#         model_ckpt: CGIPConfig.image_model_pretrained,
#         device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
#         trainable=CGIPConfig.image_model_trainable
#     ):
#         super(Img2MolInference, self).__init__()
#         self.device = device
#         print("Initializing Img2Mol Model with random weights.")
#         self.model = Img2MolPlModel(trainable=trainable)
#         if model_ckpt is not None:
#             print(f"Loading checkpoint: {model_ckpt}")
#             self.model = self.model.load_from_checkpoint(model_ckpt)



if __name__ == "__main__":
    pl_model = Img2Mol()
    print(pl_model)



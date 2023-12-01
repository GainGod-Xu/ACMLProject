import torch
from torch import nn
import numpy as np
from transformers import RobertaModel
from transformers import RobertaTokenizer
from transformers import RobertaConfig
from Utils.CMRPConfig import CMRPConfig

# Modified from https://github.com/Qihoo360/CReSS/blob/master/model/model_smiles.py

class SmilesModel(nn.Module):
    def __init__(self,
                 roberta_tokenizer_path=None,
                 smiles_maxlen=300,
                 vocab_size=181,
                 max_position_embeddings=505,
                 num_attention_heads=12,
                 num_hidden_layers=6,
                 type_vocab_size=1,
                 feature_dim=768,
                 device=CMRPConfig.device,
                 **kwargs
                 ):
        super(SmilesModel, self).__init__(**kwargs)
        self.smiles_maxlen = smiles_maxlen
        self.feature_dim = feature_dim
        # self.smiles_tokenizer = RobertaTokenizer.from_pretrained(
        #         roberta_tokenizer_path, max_len=self.smiles_maxlen)
        #self.device = device
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            hidden_size=self.feature_dim
        )

        self.model = RobertaModel(config=self.config)
        self.dense = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, input):
        hidden_states = self.model(input[:,:,0], input[:,:,1])[0][:, 0]
        features = self.dense(hidden_states)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

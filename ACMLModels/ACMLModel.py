from torch import nn
import torch.nn.functional as F
from ACMLModels.Encoder import Encoder
from ACMLModels.Projection import Projection
class ACMLModel(nn.Module):
    def __init__(
        self,
        mr1_model,
        mr2_model,
        CMRPConfig
    ):
        super().__init__()
        self.mr1_encoder = Encoder(mr1_model, CMRPConfig.mr1_model_trainable)
        self.mr2_encoder = Encoder(mr2_model, CMRPConfig.mr2_model_trainable)
        self.mr1_name = CMRPConfig.mr1_name
        self.mr2_name = CMRPConfig.mr2_name
        self.temperature = CMRPConfig.temperature
        self.device = CMRPConfig.device
        self.batch_size = CMRPConfig.batch_size
        self.mr1_projection = Projection(CMRPConfig.mr1_model_embedding, CMRPConfig.projection_dim, CMRPConfig.dropout)
        self.mr2_projection = Projection(CMRPConfig.mr2_model_embedding, CMRPConfig.projection_dim, CMRPConfig.dropout)

    def forward(self, batch):

        mr1_features = self.mr1_encoder(batch[self.mr1_name].to(self.device))
        mr2_features = self.mr2_encoder(batch[self.mr2_name].to(self.device))

        # Getting post-Embeddings of mr1 and mr2 from Projection (mandatory with same dimension)
        mr1_embeddings = self.mr1_projection(mr1_features)
        mr2_embeddings = self.mr2_projection(mr2_features)

        # Calculating the Loss
        logits = (mr1_embeddings @ mr2_embeddings.T) / self.temperature
        mr1_similarity = mr1_embeddings @ mr1_embeddings.T
        mr2_similarity = mr2_embeddings @ mr2_embeddings.T
        targets = F.softmax(
            (mr1_similarity + mr2_similarity) / 2 * self.temperature, dim=-1
        )
        mr1_loss = cross_entropy(logits, targets, reduction='none')
        mr2_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (mr1_loss + mr2_loss)/2  # shape: (batch_size)
        return loss.mean(), mr1_embeddings, mr2_embeddings

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

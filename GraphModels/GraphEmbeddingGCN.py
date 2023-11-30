
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, GlobalAttention, Set2Set

class GraphEmbeddingGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEmbeddingGCN, self).__init__()
        # convert categorical variables to one hot embeddings.
        self.embed_atom = nn.Embedding(120, hidden_dim)
        self.embed_charge = nn.Linear(1, hidden_dim)
        self.embed_chiral = nn.Embedding(10, hidden_dim)
        self.embed_hybridization = nn.Embedding(10, hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        

    def forward(self, batch):
        x = batch.x # todo: convert atomic number to one hot embeddings. How to handle hybridization?
        # print(x.max(dim=0))
        # print(x.min(dim=0))
        # x = self.embed_atom(x[:,0].long()) + self.embed_charge(x[:,1].unsqueeze(1)) + self.embed_chiral(x[:,2].long()) + self.embed_hybridization(x[:,3].long())
        x = self.embed_atom(x[:, 0].long())
        edge_index = batch.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        graph_embedding = global_add_pool(x, batch.batch)  # Graph-level pooling (global mean pooling)
        return graph_embedding


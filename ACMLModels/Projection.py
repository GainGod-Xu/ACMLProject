from torch import nn

class Projection_cell(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim,
            projection_dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(projection_dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Projection(nn.Module):
    '''
    the length of projection dim should equal to projection layers
    '''
    def __init__(
            self,
            embedding_dim,
            projection_dims,
            projection_dropout
    ):
        super().__init__()
        self.projection_module = []
        self.projection_module.append(Projection_cell(embedding_dim, projection_dims[0], projection_dropout))
        for i in range(1, len(projection_dims)):
            self.projection_module.append(Projection_cell(projection_dims[i-1], projection_dims[i], projection_dropout))

        self.projection_module = nn.Sequential(*self.projection_module)

    def forward(self, x):
        x = self.projection_module(x)
        return x



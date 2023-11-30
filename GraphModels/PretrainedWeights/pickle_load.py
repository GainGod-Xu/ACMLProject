import torch
from torch import nn
from GraphModels.PretrainedWeights.CMPNN_from_KANO import CMPNN
import numpy as np

class CMPN(nn.Module):
    def __init__(self,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(CMPN, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.graph_input = graph_input
        self.encoder = CMPNN(self.atom_fdim, self.bond_fdim)

    def forward(self, batch=None, features_batch: list[np.ndarray] = None) -> torch.FloatTensor:

        output = self.encoder.forward(batch, features_batch)
        return output


def load_pkl():
    # Create an instance of CMPNEncoder
    atom_fdim = 133  # Specify the atom feature dimension
    bond_fdim = 147  # Specify the bond feature dimension
    cmpn_encoder = CMPN(atom_fdim, bond_fdim)

    # Load the saved model using pickle
    model_path = "/home/haoxu/Documents/GitHub/CMRPProject/GraphModels/PretrainedWeights/original_CMPN_0623_1350_14000th_epoch.pkl"

    saved_model = torch.load(model_path)



    # Load the state dict of the saved model into CMPNEncoder
    cmpn_encoder.load_state_dict(saved_model)
    # Save the updated encoder part
    torch.save(cmpn_encoder.encoder.state_dict(), 'cmpnn.pth')

    # Set the CMPNEncoder in evaluation mode
    #cmpn_encoder.eval()

def load_pth():
    # Create an instance of CMPNN
    atom_fdim = 133  # Specify the atom feature dimension
    bond_fdim = 147  # Specify the bond feature dimension
    cmpnn = CMPNN(atom_fdim, bond_fdim)

    # Load the saved model using pickle
    model_path = "cmpnn.pth"

    saved_model = torch.load(model_path)

    # Load the state dict of the saved model into CMPNEncoder
    cmpnn.load_state_dict(saved_model)

    # Set the CMPNEncoder in evaluation mode
    #cmpn_encoder.eval()



if __name__ == "__main__":
    #load_pkl()
    load_pth()

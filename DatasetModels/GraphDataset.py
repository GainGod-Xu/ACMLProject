from torch.utils.data import Dataset, DataLoader
import torch
import networkx as nx
from torch_geometric.data import Data,Batch
import json
import pandas as pd
import pickle

class GraphDataset(Dataset):
    def __init__(self, files, CMRPConfig, data_loading_function='gdata_loading_method2'):
        self.files = files
        self.graphs_path = CMRPConfig.graphs_path
        self.data_loading_function_name = data_loading_function

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.files[idx]

    def __getitem__(self, idx):
        graph_path = self.graphs_path + self.files[idx]
        graph_data = None

        if self.data_loading_function_name == 'gdata_loading_method1':
            # loaa from json file
            graph_data = self.gdata_loading_method1(graph_path)
        elif self.data_loading_function_name == 'gdata_loading_method2':
            # load from pickle file
            graph_data = self.gdata_loading_method2(graph_path)
        return graph_data

    ### Methods to load molecular graphs###
    ## Method1 loads molecular graphs in json data
    @staticmethod
    def gdata_loading_method1(graph_path):
        # Create an empty graph
        graph = nx.Graph()

        # Load the JSON data from the file
        with open(graph_path, 'r') as file:
            json_data = json.load(file)

        # Add atoms as nodes to the graph
        atoms = json_data["atoms"]
        for i, atom in enumerate(atoms):
            graph.add_node(i, **atom)

        # Add bonds as edges to the graph
        bonds = json_data["bonds"]
        for bond in bonds:
            begin_idx = bond["begin_atom_idx"]
            end_idx = bond["end_atom_idx"]
            graph.add_edge(begin_idx, end_idx, **bond)

        # Convert the graph to a PyTorch Geometric Data object
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()

        # add bond type as edge attribute
        edge_attr = torch.tensor([graph.edges[edge]['bond_type'] for edge in graph.edges()], dtype=torch.long).unsqueeze(-1)

        x = torch.tensor([[float(item) for item in list(graph.nodes[node].values())] for node in graph.nodes()], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def gdata_loading_method2(graph_path):

        # load graph from pickle file
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)

        return data

    ###Collate function: collect and pad the batch data into same dims/shapes###
    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(batch)



###Test Unit###
def main():
    # Define graph files
    df = pd.read_csv(f"{CMRPConfig.dataset_path}/dataset.csv")
    graph_files = df['graph']
    # Create GraphDataset instance
    dataset = GraphDataset(graph_files)

    # Create a data loader with custom collate function
    batch_size = 10
    shuffle = True
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=GraphDataset.collate_fn,drop_last=CGIPConfig.drop_last)

    # Iterate over the data loader
    for batch in data_loader:
        # Process the batch as needed
        #print("batch size", batch.size)
        print("batch x size", batch['x'].shape)
        print("batch edge_index size", batch['edge_index'].shape)



if __name__ == "__main__":
    main()

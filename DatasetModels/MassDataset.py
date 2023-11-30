import pickle
import torch
from torch.utils.data import Dataset
import gzip
import pandas as pd
class MassDataset(Dataset):
    def __init__(self, files, CMRPConfig, data_loading_function='1'):
        self.files = files
        self.mass_path = CMRPConfig.mass_path
        self.device = CMRPConfig.device
        self.data_loading_function = data_loading_function

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.files[idx]

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = self.mass_path + file_name

        intensity_tensor =None
        if self.data_loading_function == '1':
            intensity_tensor = self.mdata_loading_method1(file_path)
        elif self.data_loading_function == '2':
            intensity_tensor = self.mdata_loading_method2(file_path)
        else:
            pass

        return intensity_tensor.unsqueeze(0)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)

    @staticmethod
    def mdata_loading_method1(file_path):
        with gzip.open(file_path, 'rb') as file:
            intensity = pickle.load(file)
        intensity_tensor = torch.tensor(intensity)
        return intensity_tensor

    @staticmethod
    def mdata_loading_method2(file_path):
        df = pd.read_csv(file_path)
        # Drop the 'Mass' column
        df = df.drop('Mass', axis=1)
        # Convert the 'Intensity' column to a PyTorch tensor
        intensity_tensor = torch.tensor(df['Intensity'].values)
        return intensity_tensor

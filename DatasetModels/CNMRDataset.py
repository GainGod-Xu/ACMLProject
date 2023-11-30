import torch
from torch.utils.data import Dataset
import numpy as np

#old dataset class for old encoding
class CNMRDataset(Dataset):
    def __init__(self, files, CMRPConfig):
        self.files = files
        self.CNMR_path = CMRPConfig.CNMR_path
        self.device = CMRPConfig.device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.CNMR_path + self.files[index]

        # Load NMR data from file
        with open(file_path, 'r') as file:
            nmr_text = file.read().strip().split(',')
            nmr = [float(value.strip()) for value in nmr_text]

        # Preprocess NMR data
        item = self.preprocess_nmr(nmr)

        return item.to(self.device)

    def collate_fn(self, batch):
        # Get the data from each sample in the batch
        data = [item for item in batch]
        # Stack the data along a new dimension (batch dimension)
        data = torch.stack(data, dim=0)
        return data

    @staticmethod
    def preprocess_nmr(nmr, scale=10, min_value=-50, max_value=350):
        units = (max_value - min_value) * scale
        item = np.zeros(units)
        nmr = [round((value - min_value) * scale) for value in nmr]
        for index in nmr:
            if index < 0:
                item[0] = 1
            elif index >= units:
                item[-1] = 1
            else:
                item[index] = 1
        item = torch.from_numpy(item).to(torch.float32)
        return item



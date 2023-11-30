import torch
from torch.utils.data import Dataset
import os
import nmrglue as ng
import numpy as np

class HNMRDataset(Dataset):
    def __init__(self, files, CMRPConfig):
        self.files = files
        self.HNMR_path = CMRPConfig.hnmr_path
        self.device = CMRPConfig.device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.HNMR_path, self.files[index])

        dic, data = ng.jcampdx.read(file_path)

        # Extract the spectrum data
        min_x = float(dic['MINX'][0])
        max_x = float(dic['MAXX'][0])
        npoints = int(dic['NPOINTS'][0])

        # cap and fill x values between 0 - 12
        step = (max_x - min_x) / npoints

        if 0 < min_x:  # add to right
            r_to_add = round(min_x / step)
            data2 = np.append(data, np.zeros(r_to_add))
        else:
            r_to_skip = round(- min_x / step)
            data2 = data[: len(data) - r_to_skip]
        if 12 > max_x:
            l_to_skip = 0
            l_to_add = round((12 - max_x) / step)
            data2 = np.append(np.zeros(l_to_add), data2)
        else:
            l_to_add = 0
            l_to_skip = round((max_x - 12) / step)
            data2 = data2[l_to_skip:]

        # normalize
        data2 = data2 / np.max(data2)
        return torch.from_numpy(data2).unsqueeze(0) #, self.files[index]

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.files[idx]

    def collate_fn(self, batch):
        # Get the data from each sample in the batch
        data = [item for item in batch]
        # Stack the data along a new dimension (batch dimension)
        data = torch.stack(data, dim=0)
        return data
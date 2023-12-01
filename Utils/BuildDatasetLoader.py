from DatasetModels.ACMLDataset import ACMLDataset
from DatasetModels.GraphDataset import GraphDataset
from DatasetModels.ImageDataset import ImageDataset
from DatasetModels.CNMRDataset import CNMRDataset
from DatasetModels.SmilesDataset import SmilesDataset
from DatasetModels.HNMRDataset import HNMRDataset
from DatasetModels.MassDataset import MassDataset
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader



def build_dataset_loader(dataframe, CMRPConfig):

    # Load pars from CMRPConfig
    mr1_name = CMRPConfig.mr1_name
    mr2_name = CMRPConfig.mr2_name
    random_seed = CMRPConfig.random_seed
    validation_ratio = CMRPConfig.validation_ratio
    batch_size = CMRPConfig.batch_size
    shuffle = CMRPConfig.shuffle
    drop_last = CMRPConfig.drop_last

    # Create mr1_dataset and mr2_dataset instances
    dataset_mapping = {
        'graph': GraphDataset,
        'image': ImageDataset,
        'cnmr': CNMRDataset,
        'hnmr': HNMRDataset,
        'smiles': SmilesDataset,
        'mass': MassDataset
    }

    mr1_dataset = dataset_mapping.get(mr1_name)(dataframe[mr1_name], CMRPConfig)
    mr2_dataset = dataset_mapping.get(mr2_name)(dataframe[mr2_name], CMRPConfig)

    # Create CGIPDataset instance
    cmrp_dataset = ACMLDataset(mr1_dataset, mr2_dataset, CMRPConfig)


    # Get the total number of samples in the dataset
    total_samples = len(cmrp_dataset)

    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random indices for splitting the dataset
    indices = np.random.permutation(total_samples)

    # Calculate the number of samples for validation
    num_valid_samples = int(total_samples * validation_ratio)

    # Split the indices into training and validation sets
    valid_indices = indices[:num_valid_samples]
    train_indices = indices[num_valid_samples:]

    # Create the training and validation datasets using the selected indices
    train_dataset = Subset(cmrp_dataset, train_indices)
    valid_dataset = Subset(cmrp_dataset, valid_indices)

    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=cmrp_dataset.collate_fn)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=cmrp_dataset.collate_fn)
    return train_dataset_loader, valid_dataset_loader


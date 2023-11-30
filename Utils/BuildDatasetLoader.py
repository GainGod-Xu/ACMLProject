from DatasetModels.CMRPDataset import CMRPDataset
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
    cmrp_dataset = CMRPDataset(mr1_dataset, mr2_dataset, CMRPConfig)

    # load saved data list
    # import torch
    # cmrp_dataset_saved = torch.load("./data_25W.pt")
    # print(len(cmrp_dataset_saved))


    # import torch
    # from tqdm import tqdm
    # data_list = []
    # data_list = [cmrp_dataset[i] for i in tqdm(range(len(cmrp_dataset)))]
    # print(len(data_list))
    # torch.save(data_list, "data_25W.pt")

    # print("begin filtering...")
    # # will remove later
    # idx = []
    # for i in range(len(mr2_dataset)):
    #     if (mr2_dataset[i].x.shape[0] <= 2) or (mr2_dataset[i].edge_index.shape[0] < 2) or (
    #             mr2_dataset[i].x.shape[0] > 50):
    #         idx.append(i)
    # print(len(idx))
    # ddf = dataframe.drop(idx)
    # ddf.to_csv("dataset_image_27W_filter_50.csv")
    #
    # idx = []
    # for i in range(len(mr2_dataset)):
    #     if (mr2_dataset[i].x.shape[0] <= 2) or (mr2_dataset[i].edge_index.shape[0] < 2) or (
    #             mr2_dataset[i].x.shape[0] > 100):
    #         idx.append(i)
    # print(len(idx))
    # ddf = dataframe.drop(idx)
    # ddf.to_csv("dataset_image_27W_filter_100.csv")

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
    # train_dataset = Subset(cmrp_dataset_saved, train_indices)
    # valid_dataset = Subset(cmrp_dataset_saved, valid_indices)

    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=cmrp_dataset.collate_fn)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=cmrp_dataset.collate_fn)
    return train_dataset_loader, valid_dataset_loader


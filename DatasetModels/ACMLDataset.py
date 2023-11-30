from torch.utils.data import Dataset
class ACMLDataset(Dataset):
    def __init__(self, mr1_dataset, mr2_dataset, CMRPConfig):
        self.mr1_dataset = mr1_dataset
        self.mr2_dataset = mr2_dataset
        self.mr1 = CMRPConfig.mr1_name
        self.mr2 = CMRPConfig.mr2_name

    def __len__(self):
        return len(self.mr1_dataset)

    def __getitem__(self, idx):
        mr1_data = self.mr1_dataset[idx]
        mr2_data = self.mr2_dataset[idx]
        mr1_file_name = self.mr1_dataset.get_sample_name(idx)
        mr2_file_name = self.mr2_dataset.get_sample_name(idx)
        return {self.mr1: mr1_data, self.mr2: mr2_data, self.mr1 + '_filename': mr1_file_name, self.mr2 + '_filename': mr2_file_name}

    def collate_fn(self, batch):
        mr1_batch = [item[self.mr1] for item in batch]
        mr1_batch = self.mr1_dataset.collate_fn(mr1_batch)

        mr2_batch = [item[self.mr2] for item in batch]
        mr2_batch = self.mr2_dataset.collate_fn(mr2_batch)

        mr1_names = [item[self.mr1 + '_filename'] for item in batch]  # Extract sample names
        mr2_names = [item[self.mr2 + '_filename'] for item in batch]  # Extract sample names
        return {self.mr1: mr1_batch, self.mr2: mr2_batch, self.mr1 + '_names': mr1_names, self.mr2 + '_names': mr2_names}





from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch

# For smiles tokenization, please refer to: https://github.com/Qihoo360/CReSS/blob/master/model/model_smiles.py


class SmilesDataset(Dataset):
    def __init__(self, files, CMRPConfig):
        self.files = files
        self.smiles_path = CMRPConfig.smiles_path
        self.tokenizer_path = CMRPConfig.mr1_model_tokenizer
        self.smiles_tokenizer = RobertaTokenizer.from_pretrained(
            self.tokenizer_path, max_len=300)

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.files[idx]
    
    def __getitem__(self, index):
        file_path = self.smiles_path + self.files[index]

        # Load Smiles data from file
        with open(file_path, 'r') as file:
            smiles_str = file.read()

        # Tokenization
        encode_dict = self.smiles_tokenizer.encode_plus(
            text=smiles_str,
            max_length=300,
            padding='max_length',
            truncation=True)

        return encode_dict

    @staticmethod
    def collate_fn(batch):
        smiles_ids = []
        smiles_mask = []
        for dic in batch:
            smiles_ids.append(dic['input_ids'])
            smiles_mask.append(dic['attention_mask'])

        smiles_ids = torch.tensor(smiles_ids)
        smiles_mask = torch.tensor(smiles_mask)

        return torch.stack([smiles_ids, smiles_mask], dim=2)

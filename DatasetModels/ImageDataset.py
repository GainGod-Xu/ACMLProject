from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_files, CMRPConfig, data_loading_method='idata_loading_method2'):
        self.image_files = image_files
        self.images_path = CMRPConfig.images_path
        self.data_loading_method_name = data_loading_method

    def __len__(self):
        return len(self.image_files)

    def get_sample_name(self, idx):
        # Assuming you have a list of sample names in the same order as the dataset
        return self.image_files[idx]

    def __getitem__(self, idx):
        image_path = self.images_path + self.image_files[idx]
        image_tensor = None

        if self.data_loading_method_name == 'idata_loading_method1':
            image_tensor = self.idata_loading_method1(image_path)
            #image_tensor = image_tensor.unsqueeze(0)
        elif self.data_loading_method_name == 'idata_loading_method2':
            image_tensor = self.idata_loading_method2(image_path)
        else:
            raise ValueError("Unrecognized image loading method.")
        return image_tensor
    
    @staticmethod
    def collate_fn(batch):
        batch_tensor = torch.stack(batch, dim=0)  # Convert image_batch to a tensor
        return batch_tensor

    ###General Loading method for images
    @staticmethod
    def idata_loading_method1(image_path, image_size=224):
        # Use for timm model
        image = Image.open(image_path).convert("RGB")

        # Define the transformations
        transform = A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

        # Apply the transformations
        transformed_image = transform(image=np.array(image))["image"]

        # Convert the image to a PyTorch tensor
        tensor = transforms.ToTensor()(transformed_image)

        return tensor

    @staticmethod
    def idata_loading_method2(image_path):
        r""" Modified from the paper `"Img2Mol – accurate SMILES recognition from molecular graphical depictions"
            <https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc01839f>`_ paper

            Original implementation: https://github.com/bayer-science-for-a-better-life/Img2Mol/tree/main.
            
            Use for cddd model.
            """
        extension = image_path.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        
        img = Image.open(image_path, "r")

        if img.mode == "RGBA":
            bg = Image.new('RGB', img.size, (255, 255, 255))
            # Paste image to background image
            bg.paste(img, (0, 0), img)
            img =  bg.convert('L')
        else:
            img = img.convert('L')

        # fit image
        old_size = img.size
        desired_size = 224
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.BICUBIC)
        new_img = Image.new("L", (desired_size, desired_size), "white")
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))

        new_img = ImageOps.expand(new_img, int(np.random.randint(5, 25, size=1)), "white")
        
        # transform image
        img_PIL = transforms.Resize((224, 224), interpolation=3)(new_img)
        img_PIL = ImageOps.autocontrast(img_PIL)
        tensor = transforms.ToTensor()(img_PIL) # size [1, 224, 224]
        
        return tensor


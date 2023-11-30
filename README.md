# ACML (Asymmetric Contrastive Multimodal Learning for Advancing Chemical Understanding)

## 1. CMRPModels Packages

### 1) CMRPPModel Class

It is a composition of GraphEncoder, ImageEncoder, ProjectionHead,
along with the definition of loss function and optimization algo.

Loss function: https://github.com/moein-shariatnia/OpenAI-CLIP

### 2) Encoder Class

It acts as an interface to allow you to load any models for any representation (mr1 and mr2) simply.

### 3) ProjectionHead Class

It projects mr1_features and mr2_features into the same vector space,
resulting in the actual mr1_embeddings and mr2_embeddings.

## 2.GraphModels


### 1) GNN model 
A general framework of GNN. Input atom feature includes atomic number and chirality tag,
while input bond feature includes bond type and bond stereo direction.

Convolution options: `gcn, gin, gat, graphsage, nnconv`. \
Neighborhood aggregation options: `add, mean, max`. \
Jump knowledge (residual connection) options: `last, max, sum`. \
Global pooling options: `add, mean, max, attention`.

## 3.ImageModels

### 1) Img2Mol Model

The file `cddd.py` included the implementation of [img2mol](https://github.com/bayer-science-for-a-better-life/Img2Mol/tree/main),
an effective image encoder for molecular images.
Before using the pretrained img2mol, download the pretrained parameters
for the default model (~2.4GB) from the [link](https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view).
Them move the downloaded file `model.ckpt` into the `ImageModels/` directory.

### 2) XXX Model for HNMR spectra (Yunrui)

## 4. SequenceModels

### 1) SmilesModel 
https://github.com/Qihoo360/CReSS/tree/master

### 2) CNMRModel 

## 5.DatasetModels

### 1) CMRPDataset 

### 2) GraphDataset 

### 3) ImageDataset

### 4) SmilesDataset 

### 5) CNMRDataset 

### 6）HNMRDataset

## 6.Datasets 

### 1) Structural_Image_Dataset


### 2) CNMR_Dataset

### 3) Smiles_Dataset


### 4) HNMR_Dataset

## 7.Utils (Everyone)

### 1) CMRPConfig

It simplifies the way that you modify variables regarding your model.

### 2) BuildDatasetLoader

It takes dataframe(containing all the files), return train_dataset_loader
and valid_dataset_loader

### 3) AvgMeter

It helps to  keep track of average loss over a period of time and learning_rate,
during the training or evaluation of a model.

### 4) TrainEpoch

It defines how the model get trained.

### 5) ValidEpoch

It defines how the model get validated.

### 6)Graph2ImageMatch

It takes random graphs from the dataset, and find the top1, top3, top10 matched images,
and calculate the corresponding accuracies.

### 7)Image2GraphMatch

It takes random images from the dataset, and find the top1, top3, top10 matched graphs,
and calculate the corresponding accuracies.

## 8. Environment Setup
The file `packages_install_handbook.txt` for hands-on instructions of installing 
packages on hpcc, using cuda=11.7.




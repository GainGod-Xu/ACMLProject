# ACML (Asymmetric Contrastive Multimodal Learning for Advancing Chemical Understanding)


## Requirements and Installation
### 1. Create virtual environment
```
conda create -n acml python=3.9 
conda activate acml
```

### 2. Install pytorch with CUDA 11.7
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 
pip install torch_geometric 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html 
pip install pytorch_lightning 
pip install -U albumentations 
pip install timm 
pip install rdkit-pypi 
pip install pandas 
pip install nmrglue 
pip install transformers
```

## Running ACML Framework
Training and evaluation of ACML are done by running the `main.py` file. In particular, a configuration file (in "Utils" 
folder) must be specified for an ACML model, containing all model, training, and evaluation parameters.
To run an experiment, the following parameters need to be specified:

- `type` : type of GNN network to use
- `num_layer`: GNN layers
- `embed_dim`: embedding dimension for input modality
- `proj_dim`: projection dimension for ACML model
- `outdir`: path to save output models

Finally, you can run an experiment
with `python main.py type=<type> num_layer=<num_layer> embed_dim=<embed_dim> proj_dim=<proj_dim>`,
where angle brackets represent a parameter value. The trained model will be saved in `outdir`.

## File structure
### 1. CMRPModels
-- CMRPPModel Class

It is the overarching framework of this project that composites of Encoder, ProjectionHead,
along with the definition of loss function and optimization algo.

Loss function: https://github.com/moein-shariatnia/OpenAI-CLIP

-- Encoder Class

It acts as an interface to allow you to load any models for any representation (mr1 and mr2) simply.

-- ProjectionHead Class

It projects mr1_features and mr2_features into the same vector space,
resulting in the actual mr1_embeddings and mr2_embeddings.

### 2. DatasetModels
The individual dataloader class for each modality. 

### 3. Datasets
The location of each dataset used in this work. It can be provided upon request. 


### 4. GraphModels
A general framework of GNN. Input atom feature includes atomic number and chirality tag,
while input bond feature includes bond type and bond stereo direction.

Convolution options: `gcn, gin, gat, graphsage, nnconv`. \
Neighborhood aggregation options: `add, mean, max`. \
Jump knowledge (residual connection) options: `last, max, sum`. \
Global pooling options: `add, mean, max, attention`.

### 5.ImageModels

Potential models to use for the "Image" modality. The file `cddd.py` included the implementation of [img2mol](https://github.com/bayer-science-for-a-better-life/Img2Mol/tree/main),
an effective image encoder for molecular images. Before using the pretrained img2mol, download the pretrained parameters
for the default model (~2.4GB) from the [link](https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view).
Then move the downloaded file `model.ckpt` into the `ImageModels/` directory.

### 6. SequenceModels
Modalities of Smiles, CNMR, HNMR, LCMass, GCMass are all considered sequence models. The respective implementations and 
pre-trained weights (if applicable) are saved here.

### 7.Utils 
Utility functions for training purposes. The Config file should be modified accordingly with each input modality.

-- CMRPConfig

It simplifies the way that you modify variables regarding your model.

-- BuildDatasetLoader

It takes dataframe(containing all the files), return train_dataset_loader
and valid_dataset_loader

-- AvgMeter

It helps to  keep track of average loss over a period of time and learning_rate,
during the training or evaluation of a model.

-- TrainEpoch

It defines how the model get trained.

-- ValidEpoch

It defines how the model get validated.

--mr2mr

It evaluates the highest similarity scores in a given batch and derives top accuracies.

## Citation 

If you use this code or its corresponding [paper](https://arxiv.org/abs/2311.06456), please cite our work as follows:

```
@article{xu2023asymmetric,
  title={Asymmetric Contrastive Multimodal Learning for Advancing Chemical Understanding},
  author={Xu, Hao and Wang, Yifei and Li, Yunrui and Hong, Pengyu},
  journal={arXiv preprint arXiv:2311.06456},
  year={2023}
}
```


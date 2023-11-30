import torch

class CMRPConfig:
    debug = True
    project_path = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    validation_ratio = 0.2
    random_seed = 4
    shuffle = True
    drop_last = True
    dataset_path = "./Datasets/HNMR_Dataset/"
    #mr1
    hnmr_path = "/HNMRdata/jdx"
    #mr2
    graphs_path = "./Datasets/HNMR_Dataset/graphs/"

    # Parameters for training
    lr = 1e-3
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 100
    image_size = 224
    acc_seq = [1, 2, 3, 5, 10, 20]

    # Parameters for Molecular Representation 1 Model
    mr1_name = "hnmr"
    mr1_model_name = 'hnmrconv'
    mr1_data_name = 'data_34k'
    mr1_model_fil = 32
    mr1_model_kernel = 7
    mr1_model_embedding = 32769 #512
    mr1_model_pretrained = True
    mr1_model_trainable = False

    # Parameters for Molecular Representation 2 Model
    mr2_name = "graph"
    mr2_model_name = "GCN"
    mr2_model_num_node_features = 4
    mr2_model_hidden_channels = 256
    mr2_model_embedding = 64
    mr2_model_pretrained = True
    mr2_model_trainable = True
    mr2_model_ckpt = './ImageModels/model.ckpt'

    # Parameters for Projection
    num_projection_layers = 1
    projection_dim = [512, 512]
    dropout = 0.1

    # Paramerts for CMRP Model
    temperature = 1.0

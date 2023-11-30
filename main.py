import pandas as pd
from Utils.CMRPConfig import CMRPConfig
from Utils.BuildDatasetLoader import build_dataset_loader
from GraphModels.GNN import  GNN
from SequenceModels import HNMRModel
from ACMLModels.ACMLModel import ACMLModel
import torch
from Utils.TrainEpoch import train_epoch
from Utils.ValidEpoch import valid_epoch
import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default='gin', help="Model type.")
parser.add_argument("--num_layer", type=int, default=5, help="Number of layers.")
parser.add_argument("--embed_dim", type=int, default=64, help="Embed dimension.")
parser.add_argument("--proj_dim", nargs="+", type=int, default=[512, 512], help="Projection dimension after encoders.")
parser.add_argument("--outdir", type=str, default='experiments/saved_models_34k', help="path for saved models")
# Parse and print the results
args = parser.parse_args()

acc_req = [1, 2, 3, 5, 10, 20]
out_dir = args.outdir
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_name = "best34k" + "_" + args.type + "_layer" + str(args.num_layer) + \
           "_edim" + str(args.embed_dim) + "_pdim." + ".".join([str(d) for d in args.proj_dim]) \
            + "_player" + str(len(args.proj_dim))
print('Output name:%s' % out_name)


# change config so that CMPRModel will use the same setting
CMRPConfig.mr2_model_type = args.type
CMRPConfig.mr2_model_embedding = args.embed_dim
CMRPConfig.mr2_model_layer = args.num_layer
CMRPConfig.projection_dim = args.proj_dim
CMRPConfig.acc_seq = acc_req

# print(‘projection dimension: ', CMRPConfig.projection_dim) # double check

def main():
    t = time.time()
    # Load graph and image file names into a dataframe
    file_name = "all_graph_34k_filter_50.csv" #cdcl_graph_9k_filter_50
    # file_name = 'dataset_2W.csv'
    df = pd.read_csv(os.path.join(CMRPConfig.dataset_path, file_name))

    # Generate train_dataset_loader and valid_dataset_loader
    train_dataset_loader, valid_dataset_loader = build_dataset_loader(df, CMRPConfig)

    print('loading time', time.time() - t)
    t = time.time()

    graph_model = GNN(args.num_layer, args.embed_dim, JK="last", gnn_type=args.type, aggr='add', graph_pooling="add").\
        to(CMRPConfig.device)

    nmr_model = HNMRModel.ConvolutionalAutoencoder(CMRPConfig.mr1_model_fil, CMRPConfig.mr1_model_kernel, 1)
    wt = np.load('SequenceModels/PretrainedWeights/weights_4ch_4lay.npy', allow_pickle=True)
    for i, param_tensor in enumerate(nmr_model.state_dict()):
        tmp = torch.from_numpy(wt[i])
        if len(tmp.shape) > 2:
            tmp = tmp.permute([2, 1, 0])
        nmr_model.state_dict()[param_tensor].copy_(tmp)

    model = ACMLModel(nmr_model, graph_model, CMRPConfig).to(CMRPConfig.device)

    print(model)
    count_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter: %d" % count_param)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CMRPConfig.lr, weight_decay=CMRPConfig.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CMRPConfig.patience, factor=CMRPConfig.factor
    )

    step = "epoch"
    log_col_names = ['epoch'] + ['best_epoch'] + ['train_mr1_to_mr2_top_' + str(i) for i in CMRPConfig.acc_seq] + \
                    ['train_mr2_to_mr1_top_' + str(i) for i in CMRPConfig.acc_seq] + \
                    ['val_mr1_to_mr2_top_' + str(i) for i in CMRPConfig.acc_seq] + \
                    ['val_mr2_to_mr1_top_' + str(i) for i in CMRPConfig.acc_seq]

    log_df = pd.DataFrame(columns=log_col_names, dtype=object)
    best_loss = float('inf')
    best_epoch = 1

    for epoch in range(CMRPConfig.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_dataset_loader, optimizer, lr_scheduler, step, acc_req, verbose=False)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_dataset_loader, acc_req, verbose=False)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), os.path.join(out_dir, out_name + '.pt'))
            best_epoch = epoch + 1
            print("Saved Best Model!, Epoch %d" % best_epoch)

        print("\n")

        tmp = [epoch + 1] + [
            best_epoch] + train_loss.mr1_to_mr2_accuracy.tolist() + train_loss.mr2_to_mr1_accuracy.tolist() + \
              valid_loss.mr1_to_mr2_accuracy.tolist() + valid_loss.mr2_to_mr1_accuracy.tolist()
        log_df.loc[len(log_df)] = tmp
        log_df.to_csv(os.path.join(out_dir, out_name + '.csv'), index=False)
    


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

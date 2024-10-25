import warnings
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
import pickle
import argparse
from types import SimpleNamespace
from matplotlib import rcParams
from DataLoader import create_loaders
from Model import MeanTrainer, GIN, DiGCN, DiGCN_IB_Sum
import networkx as nx
import torch_geometric

# Configuration
rcParams.update({'figure.autolayout': False})
warnings.filterwarnings("ignore")


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def copy_files(src_dir, dest_dir):
    for file_name in os.listdir(src_dir):
        src_file_name = os.path.join(src_dir, file_name)
        dest_file_name = os.path.join(dest_dir, file_name)
        shutil.copy(src_file_name, dest_file_name)

def run_experiment(root_path, data="HDFS", data_seed=1213, alpha=1.0, beta=0.0, epochs=150, model_seed=0, num_layers=1, device=0,
                   aggregation="Mean", bias=False, hidden_dim=64, lr=0.1, weight_decay=1e-5, batch=64):
    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    train_loader, test_loader, num_features, train_dataset, test_dataset, raw_dataset = create_loaders(
        data_name=data, batch_size=batch, dense=False, data_seed=data_seed)

    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    model = DiGCN(nfeat=num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = MeanTrainer(model=model, optimizer=optimizer, alpha=alpha, beta=beta, device=device) if aggregation == "Mean" else None

    epochinfo = []
    for epoch in tqdm(range(epochs + 1), desc="Training Progress"):
        print(f"\nEpoch {epoch}")
        svdd_loss = trainer.train(train_loader=train_loader)
        print(f"SVDD loss: {svdd_loss}")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        print(f"ROC-AUC: {roc_auc}")
       
        TEMP = SimpleNamespace(epoch_no=epoch, dists=dists, labels=labels, ap=ap, roc_auc=roc_auc, svdd_loss=svdd_loss)
        epochinfo.append(TEMP)

    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]]) + 1
    print(f"Min SVDD, at epoch {best_svdd_idx}, AP: {epochinfo[best_svdd_idx].ap}, ROC-AUC: {epochinfo[best_svdd_idx].roc_auc}")
    print(f"At the end, at epoch {epochs}, AP: {epochinfo[-1].ap}, ROC-AUC: {epochinfo[-1].roc_auc}")

    important_epoch_info = {'svdd': epochinfo[best_svdd_idx], 'last': epochinfo[-1]}
    return important_epoch_info, train_dataset, test_dataset, raw_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='OCDiGCN:')
    parser.add_argument('--root_path', default='./', type=str, help='Root path for data and logs')
    parser.add_argument('--data', default='HDFS', help='dataset name (default: HDFS)')
    parser.add_argument('--batch', type=int, default=32, help='batch size (default: 64)')
    parser.add_argument('--data_seed', type=int, default=421, help='seed to split the inlier set into train and test (default: 1213)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train (default: 150)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='number of hidden units (default: 64)')
    parser.add_argument('--layers', type=int, default=2, help='number of hidden layers (default: 2)')
    parser.add_argument('--bias', action="store_true", default=False, help='Whether to use bias terms in the GNN.')
    parser.add_argument('--aggregation', type=str, default="Mean", choices=["Max", "Mean", "Sum"], help='Type of graph level aggregation (default: Mean)')
    parser.add_argument('--use_config', action="store_true", help='Whether to use configuration from a file')
    parser.add_argument('--config_file', type=str, default="configs/config.txt", help='Name of configuration file (default: configs/config.txt)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay constant lambda (default: 1e-4)')
    parser.add_argument('--model_seed', type=int, default=0, help='Model seed (default: 0)')
    return parser.parse_args()

def load_configurations(args):
    lrs, weight_decays, layercounts, model_seeds = [args.lr], [args.weight_decay], [args.layers], [args.model_seed]
    if args.use_config:
        with open(args.config_file) as f:
            lines = [line.rstrip() for line in f]
        for line in lines:
            words = line.split()
            if words[0] == "LR":
                lrs = [float(w) for w in words[1:]]
            elif words[0] == "WD":
                weight_decays = [float(w) for w in words[1:]]
            elif words[0] == "layers":
                layercounts = [int(w) for w in words[1:]]
            elif words[0] == "model_seeds":
                model_seeds = [int(w) for w in words[1:]]
            else:
                print("Cannot parse line: ", line)
    return lrs, weight_decays, layercounts, model_seeds

def main():
    args = parse_arguments()
    lrs, weight_decays, layercounts, model_seeds = load_configurations(args)

    MyDict = {}
    for lr in lrs:
        for weight_decay in weight_decays:
            for model_seed in model_seeds:
                for layercount in layercounts:
                    print(f"Running experiment for LR={lr}, weight decay={weight_decay}, model seed={model_seed}, number of layers={layercount}")
                    MyDict[(lr, weight_decay, model_seed, layercount)], my_train, my_test, my_raw_data = run_experiment(
                        root_path=args.root_path, data=args.data, data_seed=args.data_seed, epochs=args.epochs, model_seed=model_seed,
                        num_layers=layercount, device=args.device, aggregation=args.aggregation, bias=args.bias,
                        hidden_dim=args.hidden_dim, lr=lr, weight_decay=weight_decay, batch=args.batch)

    if args.use_config:
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        with open(f'outputs/GIN_{args.aggregation}_models_{args.data}_{args.data_seed}.pkl', 'wb') as f:
            pickle.dump(MyDict, f)

    #for item in my_raw_data:
        #print(item)
    #test1 = my_raw_data[0]
    #print("Result after model process", test1)
    g = torch_geometric.utils.to_networkx(test1, to_undirected=False)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(g)
    nx.draw(g, with_labels=True)
    plt.savefig("my_graph.png")
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    
    processed_path = os.path.join(args.root_path, f'Data/{args.data}/processed')
    raw_path = os.path.join(args.root_path, f'Data/{args.data}/Raw')
    graph_raw_path = os.path.join(args.root_path, f'Data/{args.data}/Graph/Raw')

    # Clear directories
    clear_directory(processed_path)
    clear_directory(raw_path)
    
    copy_files(graph_raw_path, raw_path)
    
    main()

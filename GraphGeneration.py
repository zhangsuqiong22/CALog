import os
import shutil
import pandas as pd
import numpy as np
import networkx as nx
import json
from tqdm import tqdm
import random
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, to_undirected
import matplotlib.pyplot as plt
import os.path as osp
from GetAdjacencyMatrix import get_undirected_adj, get_appr_directed_adj, get_second_directed_adj
import argparse

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

def load_data(root_path, data_name):
    df = pd.read_csv(os.path.join(root_path, f'Data/{data_name}/{data_name}.log_structured.csv'), sep=',')
    #df['GroupId'] = df['ParameterList'].str.extract('(blk\_[-]?\d+)', expand=False)
    if data_name == 'HDFS':
        df['GroupId'] = df['ParameterList'].str.extract('(blk\_[-]?\d+)', expand=False)
    elif data_name == 'BGL':
        df['GroupId'] = df['Node']    
    elif data_name == 'Hadoop':
        df['GroupId'] = df['Container']
    elif data_name == 'Openstack':
        df['GroupId'] = df['Component']
    elif data_name == 'Kubelet':
        df['GroupId'] = df['Component']
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")    
    
    raw_df = df[["LineId", "EventId", "GroupId", "EventTemplate"]]
    return raw_df

def load_embedding(root_path, data_name):
    with open(os.path.join(root_path, f'Data/Gloves/Results/EmbeddingDict_{data_name}.json'), 'r') as fp:
        embedding_dict = json.load(fp)
    embedding_df = pd.DataFrame.from_dict(embedding_dict)
    return embedding_df

def graph_construction(root_path, data_name, my_example_df, graph_count_index, graph_loc_index, my_node_accum, new_node_accum, embedding_df):
    G = nx.MultiDiGraph()
    event_list = list(my_example_df["EventTemplate"])
    node_list = list(dict.fromkeys(event_list))
    G.add_nodes_from(node_list)
    G.add_edges_from([(event_list[v], event_list[v+1]) for v in range(len(event_list)-1)])
    
    A = nx.adjacency_matrix(G)
    
    df_A = pd.DataFrame(columns=["row", "column"])
    row_vec = (list(A.nonzero())[0]).tolist()
    col_vec = (list(A.nonzero())[1]).tolist()
    row_vec = [a+1 for a in row_vec]
    col_vec = [a+1 for a in col_vec]
    df_A["row"] = row_vec
    df_A["column"] = col_vec
    fp_A = os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw/{data_name}_A.txt")
    np.savetxt(fp_A, df_A.values, fmt='%i', delimiter=', ')

    df_edge_weight = pd.DataFrame(columns=["edge_weight"])
    df_edge_weight["edge_weight"] = list(A.data)
    fp_edge_weight = os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw/{data_name}_edge_attributes.txt")
    np.savetxt(fp_edge_weight, df_edge_weight.values, fmt='%i', delimiter=', ')

    df_graph_indicator = pd.DataFrame(columns=["indicator"])
    df_graph_indicator["indicator"] = [graph_count_index+1]*len(new_node_accum)
    fp_graph_indicator = os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw/{data_name}_graph_indicator.txt")
    np.savetxt(fp_graph_indicator, df_graph_indicator.values, fmt='%i', delimiter=', ')

    df_label = pd.read_csv(os.path.join(root_path, f'Data/{data_name}/anomaly_label.csv'), sep=',')
    di_replace = {"Normal": 0, "Anomaly": 1}
    df_label = df_label.replace({"Label": di_replace})
    label_value = df_label.iloc[graph_loc_index]['Label']
    
    df_graph_labels = pd.DataFrame(columns=["labels"])
    df_graph_labels["labels"] = [label_value]
    fp_graph_labels = os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw/{data_name}_graph_labels.txt")
    np.savetxt(fp_graph_labels, df_graph_labels.values, fmt='%i', delimiter=', ')

    node_attr_list = []
    mylist = event_list
    for node_name in list(dict.fromkeys(mylist)):
        arr_vec = embedding_df[node_name].values.tolist()
        node_attr_list.append(arr_vec)
        
    df_node_attributes = pd.DataFrame(node_attr_list)
    fp_node_attributes = os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw/{data_name}_node_attributes.txt")
    np.savetxt(fp_node_attributes, df_node_attributes.values, fmt='%f', delimiter=', ')

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def read_tu_data(folder, prefix, adj_type):
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    if batch.dim() == 0:
        node_attributes = torch.empty((1, 0))
    else:
        node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, prefix, 'node_attributes')
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    if len(edge_index.shape) == 1:
        is_empty_index = 1
        data = Data()
        return data, is_empty_index

    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)

    x = cat([node_attributes])
    if edge_index.size(1) == 1:
        edge_attr = torch.tensor([[edge_attributes.item()]])
    else:
        edge_attr = cat([edge_attributes])

    y = read_file(folder, prefix, 'graph_labels', torch.long)
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    if adj_type == 'un':
        indices = edge_index
        features = x
        indices = to_undirected(indices)
        edge_index, edge_attr = get_undirected_adj(edge_index=indices, num_nodes=features.shape[0], dtype=features.dtype)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    elif adj_type == 'appr':
        alpha = 0.1
        indices = edge_index
        features = x
        edge_index, edge_attr = get_appr_directed_adj(alpha=alpha, edge_index=indices, num_nodes=features.shape[0], dtype=features.dtype, edge_weight=edge_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    elif adj_type == 'ib':
        alpha = 0.1
        indices = edge_index
        features = x
        edge_index, edge_attr = get_appr_directed_adj(alpha=alpha, edge_index=indices, num_nodes=features.shape[0], dtype=features.dtype, edge_weight=edge_attr)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        edge_index2, edge_attr2 = get_second_directed_adj(edge_index=edge_index, num_nodes=features.shape[0], dtype=features.dtype, edge_weight=edge_attr)
        data.edge_index2 = edge_index2
        data.edge_attr2 = edge_attr2

    return data, 0

def concat_graphs(root_path, data_name, read_graph, graph_count_index, my_node_accum, new_node_accum, adj_type):
    fp_A = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_A.txt")
    df_A = pd.DataFrame(read_graph.edge_index.numpy()).T
    df_A = df_A + my_node_accum
    with open(fp_A, "ab") as f:
        np.savetxt(f, df_A.values, fmt='%i', delimiter=', ')

    fp_edge_weight = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_edge_attributes.txt")
    df_edge_weight = pd.DataFrame(read_graph.edge_attr.numpy())
    with open(fp_edge_weight, "ab") as f:
        np.savetxt(f, df_edge_weight.values, fmt='%f', delimiter=', ')

    df_graph_indicator = pd.DataFrame(columns=["indicator"])
    df_graph_indicator["indicator"] = [graph_count_index+1]*len(new_node_accum)
    fp_graph_indicator = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_graph_indicator.txt")
    with open(fp_graph_indicator, "ab") as f:
        np.savetxt(f, df_graph_indicator.values, fmt='%i', delimiter=', ')

    fp_graph_labels = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_graph_labels.txt")
    df_graph_labels = pd.DataFrame([read_graph.y.numpy()])
    with open(fp_graph_labels, "ab") as f:
        np.savetxt(f, df_graph_labels.values, fmt='%i', delimiter=', ')

    fp_node_attributes = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_node_attributes.txt")
    df_node_attributes = pd.DataFrame(read_graph.x.numpy())
    with open(fp_node_attributes, "ab") as f:
        np.savetxt(f, df_node_attributes.values, fmt='%f', delimiter=', ')

    if adj_type == 'ib':
        fp_A2 = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_A2.txt")
        df_A2 = pd.DataFrame(read_graph.edge_index2.numpy()).T
        df_A2 = df_A2 + my_node_accum
        with open(fp_A2, "ab") as f:
            np.savetxt(f, df_A2.values, fmt='%i', delimiter=', ')

        fp_edge_weight2 = os.path.join(root_path, f"Data/{data_name}/Graph/Raw/{data_name}_edge_attributes2.txt")
        df_edge_weight2 = pd.DataFrame(read_graph.edge_attr2.numpy())
        with open(fp_edge_weight2, "ab") as f:
            np.savetxt(f, df_edge_weight2.values, fmt='%f', delimiter=', ')

def draw_sample(num_samples, anomaly_per, draw_code, all_event_df, group_to_check):
    if draw_code == 1:
        anomaly_index_list = all_event_df.index[all_event_df['Label'] == "Anomaly"].tolist()
        normal_index_list = all_event_df.index[all_event_df['Label'] == "Normal"].tolist()
        random.seed(0)
        anomal_drawn = random.sample(anomaly_index_list, int(num_samples * anomaly_per))
        normal_drawn = random.sample(normal_index_list, int(num_samples * (1 - anomaly_per)))
        sample_drawn = anomal_drawn + normal_drawn
        list_group = [group_to_check[my_idx] for my_idx in sample_drawn]
        return list_group, sample_drawn
    else:
        anomaly_index_list = all_event_df.index[all_event_df['Label'] == "Anomaly"].tolist()
        normal_index_list = all_event_df.index[all_event_df['Label'] == "Normal"].tolist()
        all_samples = anomaly_index_list + normal_index_list
        list_group = [group_to_check[my_idx] for my_idx in all_samples]
        return list_group, all_samples

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Generation from Logs')
    parser.add_argument('--root_path', required=True, help='Root path of the Logs2Graph project')
    parser.add_argument('--data_name', default='HDFS', help='Dataset name (default: HDFS)')
    parser.add_argument('--num_graphs_to_test', type=int, default=10000, help='Number of graphs to test (default: 10000)')
    parser.add_argument('--anomaly_percentage', type=float, default=0.03, help='Anomaly percentage (default: 0.03)')
    parser.add_argument('--draw_code_val', type=int, default=1, help='Draw code value (default: 1)')
    return parser.parse_args()

def main(root_path, data_name, num_graphs_to_test=10000, anomaly_percentage=0.03, draw_code_val=1):
    clear_directory(os.path.join(root_path, f'Data/{data_name}/Graph/Raw'))

    raw_df = load_data(root_path, data_name)
    embedding_df = load_embedding(root_path, data_name)

    all_event_df = pd.read_csv(os.path.join(root_path, f'Data/{data_name}/anomaly_label.csv'), sep=',')
    group_to_check = list(all_event_df["BlockId"])

    list_group, list_group_idx = draw_sample(num_graphs_to_test, anomaly_percentage, draw_code_val, all_event_df, group_to_check)

    all_event_list = []
    count_index = 0

    for group_name in tqdm(list_group):
        example_df = raw_df[raw_df["GroupId"] == group_name]
        node_accum = max(len(all_event_list) + 1, 1)
        new_event_list = list(dict.fromkeys(example_df["EventTemplate"]))

        graph_construction(root_path, data_name, example_df, count_index, list_group_idx[count_index], node_accum, new_event_list, embedding_df)

        read_graph, empty_index = read_tu_data(os.path.join(root_path, f"Data/{data_name}/Graph/TempRaw"), data_name, 'ib')

        if empty_index == 0:
            concat_graphs(root_path, data_name, read_graph, count_index, node_accum, new_event_list, 'ib')
            all_event_list += new_event_list
            count_index += 1

    all_event_list = list(dict.fromkeys(all_event_list))

if __name__ == "__main__":
    args = parse_args()
    main(args.root_path, args.data_name, args.num_graphs_to_test, args.anomaly_percentage, args.draw_code_val)

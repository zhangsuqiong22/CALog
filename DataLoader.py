import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.data import DataLoader, DenseDataLoader
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    print(path)
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    #print("----row----")
    #print(row)
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
        
    if data.x is not None:
        slices['x'] = node_slice
    else:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
        
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
            
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    if data.edge_index2 is not None:
        row2, _ = data.edge_index2
        edge_slice2 = torch.cumsum(torch.from_numpy(np.bincount(batch[row2])), 0)
        edge_slice2 = torch.cat([torch.tensor([0]), edge_slice2])
        
        data.edge_index2 -= node_slice[batch[row2]].unsqueeze(0)
        
        slices['edge_index2'] = edge_slice2
        slices['edge_attr2'] = edge_slice2

    return data, slices

def read_tu_data(folder, prefix):
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1 
    edge_index2 = read_file(folder, prefix, 'A2', torch.long).t() - 1 

    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    print(batch)
    print(batch.size(0))

    node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, prefix, 'node_attributes', torch.float32)
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)
        
    edge_attributes2 = torch.empty((edge_index2.size(1), 0))
    edge_attributes2 = read_file(folder, prefix, 'edge_attributes2')
    if edge_attributes2.dim() == 1:
        edge_attributes2 = edge_attributes2.unsqueeze(-1)

    x = cat([node_attributes])
    #print("-------------x------------")
    #print(x)

    edge_attr = cat([edge_attributes])
    edge_attr2 = cat([edge_attributes2])
  
    y = read_file(folder, prefix, 'graph_labels', torch.long)
    
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    edge_index2, edge_attr2 = coalesce(edge_index2, edge_attr2, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2
        
    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_attributes2': edge_attributes2.size(-1)
    }

    return data, slices, sizes


class ParseDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        load_data = torch.load(self.processed_paths[0])
              
        self.data, self.slices, self.sizes = load_data
        
        num_node_attributes = self.num_node_attributes
        self.data.x = self.data.x[:, :num_node_attributes]
        
        num_edge_attrs = self.num_edge_attributes
        self.data.edge_attr = self.data.edge_attr[:, :num_edge_attrs]
        
        num_edge_attrs2 = self.num_edge_attributes2
        self.data.edge_attr2 = self.data.edge_attr2[:, :num_edge_attrs2]

    @property
    def raw_dir(self) -> str:
        name = f'Raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']
    
    @property
    def num_edge_attributes2(self) -> int:
        return self.sizes['num_edge_attributes2']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
 
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


DATA_PATH = 'Data'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

def load_data(data_name, dense=False, seed=1213, save_indices_to_disk=True):
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    if os.path.exists(DATA_PATH + "/" + data_name + "/Raw/"):
        print("++++++++find dataset++++++++++++")
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)
    else:
        raise NotImplementedError
 
    dataset = dataset_raw
    dataset_list = [data for data in dataset]

    train_indices = [i for i, data in enumerate(dataset_list) if data.y.item()==0 and newcoin.random()<0.7]
    test_normal_indices = [i for i, data in enumerate(dataset_list) if i not in train_indices and data.y.item()==0 ]
    test_abnormal_indices = [i for i, data in enumerate(dataset_list) if i not in train_indices and data.y.item()==1 ]
    
    train_indices = train_indices + test_abnormal_indices[:40]
    train_dataset = [dataset_list[idx] for idx in train_indices]
    test_dataset = [dataset_list[idx] for idx in range(len(dataset_list)) if idx not in train_indices]

    return train_dataset, test_dataset, dataset_raw

def create_loaders(data_name, batch_size=64, dense=False, data_seed=1213):
    train_dataset, test_dataset, dataset_raw = load_data(data_name, dense=dense, seed=data_seed)

    print("After downsampling and test-train splitting, distribution of classes:")
    labels = np.array([data.y.item() for data in train_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TRAIN: Number of graphs: %d, Class distribution %s"%(len(train_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TEST: Number of graphs: %d, Class distribution %s"%(len(test_dataset), label_dist))

    Loader = DenseDataLoader if dense else DataLoader
    num_workers = 0
    
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, train_dataset[0].num_features, train_dataset, test_dataset, dataset_raw

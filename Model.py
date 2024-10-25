import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.metrics import average_precision_score, roc_auc_score
import torch_geometric.nn as pyg_nn

from DIGCNConv import DIGCNConv

# class MeanTrainer:
#     def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
#         self.device = device
#         self.model = model
#         self.optimizer = optimizer
#         self.center = None
#         self.reg_weight = 0
#         self.alpha = alpha
#         self.beta = beta
#         self.regularizer = regularizer   
    
#     def train(self, train_loader):
#         print("\n++++++++++++++++train()++++++++++++++++")
#         self.model.train()
        
#         if self.center == None:
#             F_list = []

#         svdd_loss_accum = 0
#         total_iters = 0

#         for batch in train_loader:
#             print("\n++++++++++++++++batch training start++++++++++++++++")
            
#             train_embeddings = self.model(batch)
#             print("++++++++++++++++batch training end++++++++++++++++")
            
#             mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings]
#             F_train = torch.stack(mean_train_embeddings)
                
#             if self.center == None:
#                 F_list.append(F_train)
#             else:
#                 train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu()
#                 svdd_loss = torch.mean(train_scores)
                
#                 self.optimizer.zero_grad()
#                 svdd_loss.backward()    
#                 self.optimizer.step()
                
#                 svdd_loss_accum += svdd_loss.detach().cpu().numpy()
#                 total_iters += 1

#         if self.center == None:
#             full_F_list = torch.cat(F_list)
#             self.center = torch.mean(full_F_list, dim=0).detach()
#             average_svdd_loss = -1
#         else:
#             average_svdd_loss = svdd_loss_accum / total_iters

#         return average_svdd_loss

#     def test(self, test_loader):
#         print("\n+++++++++++++++test()++++++++++++++++")
#         self.model.eval()
        
#         with torch.no_grad():
#             dists_list = []
#             for batch in test_loader:
#                 print("Batch info:")
#                 print("Node features (x):", batch.x)
#                 print("Edge indices (edge_index):", batch.edge_index)
#                 print("Graph labels (y):", batch.y)

#                 test_embeddings = self.model(batch)
#                 mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings]
#                 F_test = torch.stack(mean_test_embeddings)
                
#                 batch_dists = torch.sum((F_test - self.center)**2, dim=1).cpu()
#                 dists_list.append(batch_dists)
            
#             labels = torch.cat([batch.y for batch in test_loader])
#             dists = torch.cat(dists_list)

#             ap = average_precision_score(y_true=labels, y_score=dists, average=None, pos_label=1, sample_weight=None)
#             print("y_true label", labels)
#             roc_auc = roc_auc_score(y_true=labels, y_score=dists, average=None, sample_weight=None, max_fpr=None, multi_class='raise', labels=None)

#             return ap, roc_auc, dists, labels


# class GIN(nn.Module):
#     """
#     Graph Isomorphism Network (GIN) model.
#     Note: Batch normalization can prevent divergence. Take care of this later.
#     """
#     def __init__(self, nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
#         super(GIN, self).__init__()
#         self.norm = BatchNorm1d
#         self.nlayer = nlayer
#         self.act = act
#         self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
#         self.pooling = global_mean_pool
#         self.dropout = nn.Dropout(dropout)

#         self.convs = nn.ModuleList()
#         self.nns = nn.ModuleList()
#         self.bns = nn.ModuleList()

#         for i in range(nlayer):
#             self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), act, Linear(nhid, nhid, bias=bias)))
#             self.convs.append(GINConv(self.nns[-1]))
#             self.bns.append(self.norm(nhid))

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.transform(x)  # Apply initial transformation

#         for i in range(self.nlayer):
#             x = self.dropout(x)
#             x = self.convs[i](x, edge_index)
#             x = self.act(x)
#             x = self.bns[i](x)

#         emb_list = [x[data.batch == g] for g in range(data.num_graphs)]
#         return emb_list

# class DiGCN(nn.Module):
#     """
#     Directed Graph Convolutional Network (DiGCN) model.
#     """
#     def __init__(self, nfeat, nhid, nlayer, dropout=0, bias=False, **kwargs):
#         super(DiGCN, self).__init__()
#         self.conv1 = DIGCNConv(nfeat, nhid)
#         self.conv2 = DIGCNConv(nhid, nhid)

#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.0, training=self.training)
#         x = self.conv2(x, edge_index, edge_attr)

#         emb_list = [x[data.batch == g] for g in range(data.num_graphs)]
#         return emb_list

# class InceptionBlock(nn.Module):
#     """
#     Inception Block for DiGCN model.
#     """
#     def __init__(self, in_dim, out_dim):
#         super(InceptionBlock, self).__init__()
#         self.ln = Linear(in_dim, out_dim)
#         self.conv1 = DIGCNConv(in_dim, out_dim)
#         self.conv2 = DIGCNConv(in_dim, out_dim)

#     def reset_parameters(self):
#         self.ln.reset_parameters()
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()

#     def forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2):
#         x0 = self.ln(x)
#         x1 = self.conv1(x, edge_index, edge_attr)
#         x2 = self.conv2(x, edge_index2, edge_attr2)
#         return x0, x1, x2

# class DiGCN_IB_Sum(nn.Module):
#     """
#     DiGCN model with Inception Blocks and Sum aggregation.
#     """
#     def __init__(self, nfeat, nhid, nlayer, bias=False, **kwargs):
#         super(DiGCN_IB_Sum, self).__init__()
#         self.ib1 = InceptionBlock(nfeat, nhid)
#         self.ib2 = InceptionBlock(nhid, nhid)
#         self.ib3 = InceptionBlock(nhid, nhid)

#     def reset_parameters(self):
#         self.ib1.reset_parameters()
#         self.ib2.reset_parameters()
#         self.ib3.reset_parameters()

#     def forward(self, data, dropout_v=0.1):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         edge_index2, edge_attr2 = data.edge_index2, data.edge_attr2

#         x = self.apply_inception_block(self.ib1, x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v)
#         x = self.apply_inception_block(self.ib2, x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v)
#         x = self.apply_inception_block(self.ib3, x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v)

#         emb_list = [x[data.batch == g] for g in range(data.num_graphs)]
#         return emb_list

#     def apply_inception_block(self, block, x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v):
#         """
#         Apply an inception block with dropout.
#         """
#         x0, x1, x2 = block(x, edge_index, edge_attr, edge_index2, edge_attr2)
#         x0 = F.dropout(x0, p=dropout_v, training=self.training)
#         x1 = F.dropout(x1, p=dropout_v, training=self.training)
#         x2 = F.dropout(x2, p=dropout_v, training=self.training)
#         return x0 + x1 + x2

class MeanTrainer:
    def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        """
        Initializes the MeanTrainer with the model, optimizer, and SVDD parameters.
        """
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer
        self.center = None

    def train(self, train_loader):
        """
        Train the model with SVDD loss and center calculation for the first epoch.
        """
        print("Training started...")
        self.model.train()
        svdd_loss_accum = 0
        total_iters = 0
        F_list = [] if self.center is None else None

        for batch in train_loader:
            train_embeddings = self.model(batch)
            mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings]
            F_train = torch.stack(mean_train_embeddings)

            if self.center is None:
                F_list.append(F_train)
            else:
                svdd_loss = self.compute_svdd_loss(F_train)
                self.optimizer.zero_grad()
                svdd_loss.backward()
                self.optimizer.step()

                svdd_loss_accum += svdd_loss.item()
                total_iters += 1

        if self.center is None:
            self.center = torch.mean(torch.cat(F_list), dim=0).detach()
            return -1  # Return a dummy value for the first epoch
        else:
            return svdd_loss_accum / total_iters

    def compute_svdd_loss(self, F_train):
        """
        Computes the SVDD loss based on the distance from the center.
        """
        train_scores = torch.sum((F_train - self.center) ** 2, dim=1).cpu()
        return torch.mean(train_scores)

    def test(self, test_loader):
        """
        Evaluate the model using Average Precision (AP) and ROC AUC scores.
        """
        print("Testing started...")
        self.model.eval()
        dists_list, labels_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                test_embeddings = self.model(batch)
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings]
                F_test = torch.stack(mean_test_embeddings)
                dists = torch.sum((F_test - self.center) ** 2, dim=1).cpu()
                dists_list.append(dists)

            dists = torch.cat(dists_list)
            labels = torch.cat([batch.y for batch in test_loader])
            ap = average_precision_score(y_true=labels, y_score=dists)
            roc_auc = roc_auc_score(y_true=labels, y_score=dists)

        return ap, roc_auc, dists, labels


class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False):
        """
        GIN model with Batch Normalization and multiple GINConv layers.
        """
        super(GIN, self).__init__()
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), BatchNorm1d(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([GINConv(self.build_mlp(nhid, bias)) for _ in range(nlayer)])
        self.bns = nn.ModuleList([BatchNorm1d(nhid) for _ in range(nlayer)])

    def build_mlp(self, nhid, bias):
        """
        Helper function to create an MLP for GINConv.
        """
        return Sequential(Linear(nhid, nhid, bias=bias), self.act, Linear(nhid, nhid, bias=bias))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.transform(x)

        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)

        return self.get_graph_embeddings(x, data)

    def get_graph_embeddings(self, x, data):
        """
        Helper function to extract node embeddings per graph.
        """
        return [x[data.batch == g] for g in range(data.num_graphs)]    

class DiGCN(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, num_heads=8, bias=False, attention_dropout=0.1, **kwargs):
        """
        DiGCN model with attention mechanism.
        """
        super(DiGCN, self).__init__()
        self.conv1 = DIGCNConv(nfeat, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)
        self.attention = self.build_attention_layer(nhid, num_heads, attention_dropout)

    def build_attention_layer(self, nhid, num_heads, attention_dropout):
        """
        Builds the attention layer using GATConv.
        This method can be replaced or modified to use a different attention mechanism.
        """
        return pyg_nn.GATConv(nhid, nhid // num_heads, heads=num_heads, dropout=attention_dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.attention(x, edge_index)

        return self.get_graph_embeddings(x, data)

    def get_graph_embeddings(self, x, data):
        """
        Helper function to extract node embeddings per graph.
        """
        return [x[data.batch == g] for g in range(data.num_graphs)]
    
class InceptionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Inception block for DiGCN model.
        """
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2):
        """
        Forward pass for the Inception block.
        """
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(x, edge_index2, edge_attr2)
        return x0, x1, x2


class DiGCN_IB_Sum(nn.Module):
    def __init__(self, nfeat, nhid):
        """
        DiGCN model with Inception blocks and sum-based aggregation.
        """
        super(DiGCN_IB_Sum, self).__init__()
        self.ib1 = InceptionBlock(nfeat, nhid)
        self.ib2 = InceptionBlock(nhid, nhid)
        self.ib3 = InceptionBlock(nhid, nhid)

    def forward(self, data, dropout_v=0.1):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index2, edge_attr2 = data.edge_index2, data.edge_attr2

        x = self.inception_forward(x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v)
        return self.get_graph_embeddings(x, data)

    def inception_forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2, dropout_v):
        """
        Helper function to apply inception blocks with dropout.
        """
        for ib in [self.ib1, self.ib2, self.ib3]:
            x0, x1, x2 = ib(x, edge_index, edge_attr, edge_index2, edge_attr2)
            x = F.dropout(x0 + x1 + x2, p=dropout_v, training=self.training)
        return x

    def get_graph_embeddings(self, x, data):
        """
        Helper function to extract node embeddings per graph.
        """
        return [x[data.batch == g] for g in range(data.num_graphs)]

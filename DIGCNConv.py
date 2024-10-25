import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros

class DIGCNConv(MessagePassing):
    r"""
    The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache the adj matrix on first execution, 
                                and will use the cached version for further executions.
                                Please note that, all the normalized adj matrices (including undirected)
                                are calculated in the dataset preprocessing to reduce time comsume.
                                This parameter should only be set to :obj:`True` in transductive learning scenarios. 
                                (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias. 
                                (default: :obj:`True`)
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=False, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the convolution layer.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Edge indices.
            edge_attr (Tensor): Edge attributes (normalized adjacency matrix).
        """
        x = self.apply_weight(x)
        edge_index, norm = self.get_edge_index_and_norm(edge_index, edge_attr)
        return self.propagate(edge_index, x=x, norm=norm)

    def apply_weight(self, x):
        """Apply the learnable weight matrix to the input features."""
        return torch.matmul(x, self.weight)

    def get_edge_index_and_norm(self, edge_index, edge_attr):
        """
        Get the edge index and normalization factor.

        Args:
            edge_index (Tensor): Edge indices.
            edge_attr (Tensor): Edge attributes (normalized adjacency matrix).

        Returns:
            Tuple[Tensor, Tensor]: Edge index and normalization factor.
        """
        if self.cached and self.cached_result is not None:
            self.check_cached_edges(edge_index)

        if not self.cached or self.cached_result is None:
            self.cache_edge_index_and_norm(edge_index, edge_attr)

        return self.cached_result

    def check_cached_edges(self, edge_index):
        """Check if the cached edge index matches the current edge index."""
        if edge_index.size(1) != self.cached_num_edges:
            raise RuntimeError(
                f'Cached {self.cached_num_edges} number of edges, but found {edge_index.size(1)}. '
                'Please disable the caching behavior of this layer by removing the `cached=True` argument in its constructor.'
            )

    def cache_edge_index_and_norm(self, edge_index, edge_attr):
        """Cache the edge index and normalization factor."""
        self.cached_num_edges = edge_index.size(1)
        if edge_attr is None:
            raise RuntimeError(
                'Normalized adj matrix cannot be None. Please obtain the adj matrix in preprocessing.'
            )
        self.cached_result = edge_index, edge_attr

    def message(self, x_j, norm):
        """
        Compute the message to be propagated.

        Args:
            x_j (Tensor): Features of neighboring nodes.
            norm (Tensor): Normalization factor.

        Returns:
            Tensor: Message to be propagated.
        """
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        """
        Update node embeddings with the aggregated messages.

        Args:
            aggr_out (Tensor): Aggregated messages.

        Returns:
            Tensor: Updated node embeddings.
        """
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

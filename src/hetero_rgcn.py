import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.nn.modules.module import Module

class DynamicMLP(nn.Module):
    def __init__(self, in_channels, hidden_layer_size_list, output_channel, activation_function):
        super().__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to store layers        
        self.activation_function = activation_function

        if hidden_layer_size_list == None or len(hidden_layer_size_list) == 0: 
            # If no hidden layers are specified, just create a single linear layer
            self.layers.append(nn.Linear(in_channels, output_channel)) # handle the case where num_layers is 0
        else:
            # First layer            
            self.layers.append(nn.Linear(in_channels, hidden_layer_size_list[0]))            
            # Hidden layers (if any)
            for idx in range(1, len(hidden_layer_size_list)): # Start from 1 since we already added the first layer
                self.layers.append(nn.Linear(hidden_layer_size_list[idx-1], hidden_layer_size_list[idx])) 
            self.layers.append(nn.Linear(hidden_layer_size_list[-1], output_channel))
            # Last layer to output channel

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # x = self.activation_function(x) # Activation after each layer
        return x
    
class HeterogeneousRGCN(nn.Module):
    def __init__(self, in_channels_dict, mlp_hidden_channels_dict, mlp_out_emb_size, conv_hidden_channels, num_nodes_per_type,
                 num_entities, num_relations, conv_num_layers, num_bases, activation_function = F.relu, device="cuda:0"):  # Add mlp_layers_dict
        super().__init__()

        self.mlp_out = mlp_out_emb_size
        self.device = device
        self.num_nodes_per_type = num_nodes_per_type
        # Relation embeddings
        self.activation_function = activation_function
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, conv_hidden_channels[f"layer_{conv_num_layers-1}"]))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.mlp_dict = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            if in_channels == None:  ### Only if this node type does not have features        
                self.mlp_dict[node_type] = nn.Embedding( num_nodes_per_type[node_type], mlp_out_emb_size)
            else:
                ### Only if this node type have features
                self.mlp_dict[node_type] = nn.Linear(in_channels, mlp_out_emb_size) ### Use the nn.Embedding
                # self.mlp_dict[node_type] = DynamicMLP(in_channels, mlp_hidden_channels_dict[node_type], mlp_out_emb_size, activation_function) ### create an MLP 

        self.conv_layers = nn.ModuleList()
        self.layers_num = conv_num_layers

        self.conv_layers.append(RGCNConv(mlp_out_emb_size, conv_hidden_channels[f'layer_{0}'], num_entities, num_relations, num_bases))  # Assuming you have a list of num_layers for each convolution
        for idx in range(1, conv_num_layers):
            self.conv_layers.append(RGCNConv(conv_hidden_channels[f'layer_{idx-1}'], conv_hidden_channels[f'layer_{idx}'], num_entities, num_relations, num_bases))

    def forward(self, x_dict, edge_index):  # x_dict: dict of features, edge_index_dict: dict of edge indices
        x_list=[] 
        for node_type, features in x_dict.items():
            if features == None: ### Only if there are no features in input for this node type
                x_list.append(self.mlp_dict[node_type](torch.arange(0, self.num_nodes_per_type[node_type],
                                                                   dtype=torch.long, device=self.device)))  ### Use the nn.Embedding                 
            else:               ### Only if there are features for this node type
                x_list.append( self.mlp_dict[node_type](features)) ### Use the MLP
        x = torch.cat(x_list) 
        for layer in range(self.layers_num - 1): ### should insert the activation function and dropout here
            x = self.activation_function(self.conv_layers[layer](x, edge_index, self.device))
            # x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        # Execute the last RGCN layer where no relu and dropout are applied
        # Note: if the number of layer is 1 this will be the only layer executed, without relu
        x = self.conv_layers[self.layers_num - 1](x, edge_index, self.device)
        return x

    def distmult(self, embedding, triplets):
            s = embedding[triplets[:,0]]
            r = self.relation_embedding[triplets[:,1]]
            o = embedding[triplets[:,2]]
            score = torch.sum(s * r * o, dim=1)
            return score

    def score_loss(self, scores, target):
        return F.binary_cross_entropy_with_logits(scores, target)

    def reg_loss(self, embedding, triplets):
            """ Compute Schlichtkrull L2 penalty for the decoder """
            s_index, p_index, o_index = triplets.t()
            s, p, o = embedding[s_index, :], self.relation_embedding[p_index, :], embedding[o_index, :]
            return s.pow(2).mean() + p.pow(2).mean() + o.pow(2).mean()

class RGCNConv(Module):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, every_node, every_relation, num_bases, no_bias=False):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.every_node = every_node
        self.every_relation = every_relation
        self.num_bases = num_bases
        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(every_relation, num_bases))
        self.leakyrelu = nn.LeakyReLU()
        if no_bias:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.initialise_weights()

    def initialise_weights(self):
        weight_gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.basis, gain=weight_gain)
        nn.init.xavier_uniform_(self.att, gain=weight_gain)
        if self.bias != None:
            torch.nn.init.uniform_(self.bias)

    def forward(self, node_embeddings, triples, device):
        """ Perform a single pass of message propagation """
        # Apply weight decomposition
        weights = torch.einsum('rb, bio -> rio', self.att, self.basis)
        adj_indices, adj_size = stack_matrices(
            triples,
            self.every_node,
            self.every_relation)
        vals = torch.ones(adj_indices.size(0), dtype=torch.float, device=device) #.to(device)
        # Apply normalisation
        sums = sum_sparse(adj_indices, vals, adj_size, device=device)
        vals = vals / sums
        af = torch_sparse.spmm(adj_indices.T, vals, adj_size[0], adj_size[1], node_embeddings)
        af = af.view(self.every_relation, adj_size[1], self.in_channels) #(R, n, E)
        output = torch.einsum('rio, rni -> no', weights, af)
        # add bias to output
        if self.bias is not None:
            output = torch.add(output, self.bias)
        return output

def stack_matrices(triples, nodes_num, num_rels):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    """
    assert triples.dtype == torch.long
    R, n = num_rels, nodes_num
    size = (R * n, n)    
    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    # print(f" max triples1 {torch.max(triples[:, 1])}") 
    # print(f" max fr {torch.max(fr)} max fr offset {torch.max(offset)}")
    fr = offset + fr
    # print(f" max fr {torch.max(fr)} max fr offset {torch.max(fr)}")
    indices = torch.cat([fr[:, None], to[:, None]], dim=1)
    # print(indices)
    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[:, 0].max()}, {size}, {R}'
    assert indices[:, 1].max() < size[1], f'{indices[:, 1].max()}, {size}, {R}'
    return indices, size

def sum_sparse(indices, values, size, device):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """
    assert len(indices.size()) == len(values.size()) + 1
    k, _ = indices.size()
    ones = torch.ones((size[1], 1), device=device) #.to(device)
    # values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    values = torch.sparse_coo_tensor(indices=indices.t(), values=values, size=size, dtype=torch.float, device=device)
    # sums = torch_sparse.spmm(indices.T, values, size[0], size[1], ones)
    sums = torch.spmm(values, ones) #.to(device)
    sums = sums[indices[:, 0], 0]
    return sums.view(k)
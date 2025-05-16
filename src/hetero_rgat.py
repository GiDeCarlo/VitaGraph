import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch.nn.modules.module import Module
# from torcheval.metrics.functional import binary_auprc, binary_auroc
# from utils import uniform
    
class HeterogeneousRGAT(nn.Module):
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
        self.conv_layers.append(HRGATconv(mlp_out_emb_size, conv_hidden_channels[f'layer_{0}'], num_entities, num_relations, num_bases))  # Assuming you have a list of num_layers for each convolution
        for idx in range(1, conv_num_layers):
            self.conv_layers.append(HRGATconv(conv_hidden_channels[f'layer_{idx-1}'], conv_hidden_channels[f'layer_{idx}'], num_entities, num_relations, num_bases))

    def forward(self, x_dict, edge_index,change_points):  # x_dict: dict of features, edge_index_dict: dict of edge indices
        x_list=[] 
        for node_type, features in x_dict.items():
            if features == None: ### Only if there are no features in input for this node type
                x_list.append(self.mlp_dict[node_type](torch.arange(0, self.num_nodes_per_type[node_type],
                                                                   dtype=torch.long, device=self.device)))  ### Use the nn.Embedding 
            else:               ### Only if there are features for this node type
                x_list.append( self.mlp_dict[node_type](features)) ### Use the MLP

        x = torch.cat(x_list) 
        
        for layer in range(self.layers_num - 1): ### should insert the activation function and dropout here
            x = self.activation_function(self.conv_layers[layer](x, edge_index, change_points, self.device))

            # x = F.dropout(x, p = self.dropout_ratio, training = self.training)

        # Execute the last RGCN layer where no relu and dropout are applied
        # Note: if the number of layer is 1 this will be the only layer executed, without relu
        x = self.conv_layers[self.layers_num - 1](x, edge_index, change_points, self.device)
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

class HRGATconv(Module):
    def __init__(self, in_channels, out_channels, every_node, every_relation, num_bases, no_bias=False, no_attention=False):
        super(HRGATconv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.every_node = every_node
        self.every_relation = every_relation
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(every_relation, num_bases))
        
        self.leakyrelu = nn.LeakyReLU()

        if no_attention:
            self.register_parameter('attention', None)
        else:
            self.attention = nn.Parameter(torch.Tensor(every_relation, 2 * out_channels))
        if no_bias:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.initialise_weights()
    
    def initialise_weights(self):

        weight_gain = nn.init.calculate_gain('relu')

        nn.init.xavier_uniform_(self.basis, gain=weight_gain)
        nn.init.xavier_uniform_(self.att, gain=weight_gain)
        torch.nn.init.uniform_(self.bias)
        nn.init.xavier_uniform_(self.attention, gain=weight_gain)
    
    def forward(self, node_embeddings, triples, change_points, device):
        """ Perform a single pass of message propagation """
        
        # Apply weight decomposition
        weights = torch.einsum('rb, bio -> rio', self.att, self.basis) #.contiguous()
        
        # print("weights shape: ", weights.shape)
        # print("node_embeddings shape: ", node_embeddings.shape)

        Wh = torch.einsum('rio, ni -> rno', weights, node_embeddings).contiguous()
        h_prime = torch.zeros(self.every_node, self.out_channels, device=device).contiguous()   
        edge_type = torch.arange(0,self.every_relation-1, device=device,dtype=torch.int).contiguous() # shape: (n, 1)

        for rel in edge_type:

            start = change_points[rel]
            end = change_points[rel + 1]
            rel_index = triples[start:end][:, [0, 2]]

            # rel_index = triples[(triples[:, 1] == rel)][:,[0,2]]  ### get edges with this relation
            
            a = self.attention[rel] # (out*2)
            Wh_rel = Wh[rel] # no actual copy since Wh is contigous shape: (n, out)

            vals = torch.ones(rel_index.size(0), dtype=torch.float, device=device ) #.to(device) # shape: (n)

            # Apply normalisation 
            sums = sum_sparse(rel_index, vals, (self.every_node, self.every_node), device=device)  
            vals = vals / sums
            
            Wh_concat =  torch.cat((Wh_rel[rel_index[:, 0], :], Wh_rel[rel_index[:, 1],:]), dim=1).t() # shape: (2*out, edges)
                
            # compute the attention coef. for each nodes combination (edges) 
            edge_e = torch.exp(-self.leakyrelu(a[None, :].mm(Wh_concat).squeeze())) # shape: (1, out*2)@(2*out, edges)=(edges)
            
            # sum the attention coef. for each node neighbourhood (for softmax normalization factor)
            sum_row_e = torch_sparse.spmm(rel_index.T, edge_e, self.every_node, self.every_node, torch.ones(self.every_node, 1, device=device)) # shape: (n, n)@(n, 1)=(n, 1)

            # some nodes might node make a specific relation, therefore the normalization value might be 0.
            # to be sure this can't happen we change the zeros to ones
            # sum_row_e = torch.where(sum_row_e == 0., 1. , sum_row_e) old version
            sum_row_e = torch.clamp(sum_row_e, min=1.0)  # Clamp values to ensure a minimum of 1
                                                              # this is a bit more efficient than the old version
            # multiply attention and edge normalization, then compute the weighted aggregation of each node           
            h_prime = h_prime + torch_sparse.spmm(rel_index.T, edge_e*vals, self.every_node, self.every_node, Wh_rel) # shape: (n, n)@(n, out)=(n, out)
    
            # complete the attention normalization with sum_row_e
            # h_prime = h_prime / sum_row_e 

            h_prime = h_prime / torch.clamp(sum_row_e, min=1e-6)  # Ensure no division by zero or very small values
            h_prime = torch.clamp(h_prime, min=-1e6, max=1e6)  # Clamp h_prime to 
        output = h_prime

        # add bias to output
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
    fr = offset + fr
    
    indices = torch.cat([fr[:, None], to[:, None]], dim=1)

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

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
 
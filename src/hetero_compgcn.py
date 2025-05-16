import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch.nn.modules.module import Module

class CompGCNConv(Module):
    """
    Compositional Graph Convolutional Layer implementation.
    
    This layer performs message passing using compositional operators on entity
    and relation embeddings as described in CompGCN paper.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_relations: int, 
        comp_fn: str = 'mult', 
        dropout: float = 0.0, 
        bias: bool = True,
        edge_norm: bool = True
    ):
        """
        Initialize the CompGCN convolution layer.
        
        Args:
            in_channels: Input feature dimensionality
            out_channels: Output feature dimensionality
            num_relations: Number of relation types
            comp_fn: Composition function ('mult', 'sub', or 'corr')
            dropout: Dropout rate
            bias: Whether to use bias
            edge_norm: Whether to normalize edge messages by node degree
        """
        super(CompGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.comp_fn = comp_fn
        self.dropout = dropout
        self.edge_norm = edge_norm

        # Weight matrices
        self.w_loop = nn.Linear(in_channels, out_channels, bias=False)
        self.w_in = nn.Linear(in_channels, out_channels, bias=False)
        self.w_out = nn.Linear(in_channels, out_channels, bias=False)
        self.w_rel = nn.Linear(in_channels, out_channels, bias=False)

        # Relation embeddings for self-loops
        self.loop_rel = nn.Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters with appropriate distributions."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.w_loop.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_in.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_out.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_rel.weight, gain=gain)
        nn.init.xavier_uniform_(self.loop_rel, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compositional_op(self, ent: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """
        Apply compositional operation between entity and relation embeddings.
        
        Args:
            ent: Entity embeddings
            rel: Relation embeddings
            
        Returns:
            Composed embeddings
        """
        if self.comp_fn == 'mult':
            return ent * rel
        elif self.comp_fn == 'sub':
            return ent - rel
        elif self.comp_fn == 'corr':
            # Handle real-valued FFT correlation
            # dim = ent.shape[-1]
            # ent_fft = torch.fft.rfft(ent, dim=-1)
            # rel_fft = torch.fft.rfft(rel, dim=-1)
            # ent_fft_conj = torch.conj(ent_fft)
            # prod = ent_fft_conj * rel_fft
            # return torch.fft.irfft(prod, n=dim, dim=-1)
            return ent * rel
        else:
            raise ValueError(f"Unsupported composition: {self.comp_fn}")

    def forward(
        self, 
        node_emb: torch.Tensor, 
        rel_emb: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the CompGCN layer.
        
        Args:
            node_emb: Node embeddings [num_nodes, in_channels]
            rel_emb: Relation embeddings [num_relations, in_channels]  
            edge_index: Edge information [num_edges, 3] with (src, rel_type, dst)
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        s = edge_index[:, 0]  # source nodes
        r = edge_index[:, 1].long()  # relation types
        t = edge_index[:, 2]  # target nodes

        num_nodes = node_emb.size(0)
        
        # Calculate node degrees for normalization if needed
        if self.edge_norm:
            # Compute in-degree for normalization
            t_deg = torch_scatter.scatter_add(
                torch.ones_like(t, dtype=torch.float, device=t.device), 
                t, dim=0, dim_size=num_nodes
            )
            t_deg_inv = t_deg.pow(-0.5)  # inverse square root
            t_deg_inv[torch.isinf(t_deg_inv)] = 0
            
            # Compute out-degree for normalization
            s_deg = torch_scatter.scatter_add(
                torch.ones_like(s, dtype=torch.float, device=s.device), 
                s, dim=0, dim_size=num_nodes
            )
            s_deg_inv = s_deg.pow(-0.5)  # inverse square root
            s_deg_inv[torch.isinf(s_deg_inv)] = 0
        else:
            # No normalization
            t_deg_inv = s_deg_inv = None

        # Compose messages for forward edges
        comp_out = self.compositional_op(node_emb[s], rel_emb[r])
        msg_out = self.w_out(comp_out)
        
        # Apply normalization if enabled
        if self.edge_norm:
            msg_out = msg_out * s_deg_inv[s].unsqueeze(1)
            
        agg_out = torch_scatter.scatter_add(msg_out, t, dim=0, dim_size=num_nodes)
        if self.edge_norm:
            agg_out = agg_out * t_deg_inv.unsqueeze(1)

        # Compose messages for inverse edges (preprocessed in edge_index)
        comp_in = self.compositional_op(node_emb[t], rel_emb[r])
        msg_in = self.w_in(comp_in)
        
        # Apply normalization if enabled
        if self.edge_norm:
            msg_in = msg_in * t_deg_inv[t].unsqueeze(1)
            
        agg_in = torch_scatter.scatter_add(msg_in, s, dim=0, dim_size=num_nodes)
        if self.edge_norm:
            agg_in = agg_in * s_deg_inv.unsqueeze(1)

        # Self-loop messages
        loop_rel_expand = self.loop_rel.expand(num_nodes, -1)
        comp_loop = self.compositional_op(node_emb, loop_rel_expand)
        msg_loop = self.w_loop(comp_loop)

        # Combine with mean aggregation
        out = (agg_out + agg_in + msg_loop) / 3.0
        if self.bias is not None:
            out = out + self.bias
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out

class HeterogeneousCompGCN(nn.Module):
    """
    Heterogeneous Compositional Graph Convolutional Network for knowledge graphs
    with multiple node types.
    """
    def __init__(
        self, 
        in_channels_dict: Dict[str, Optional[int]], 
        mlp_out_emb_size: int,
        conv_hidden_channels: Dict[str, int], 
        num_nodes_per_type: Dict[str, int],
        num_entities: int, 
        num_relations: int, 
        conv_num_layers: int, 
        opn: str = "sub", 
        dropout: float = 0.1,
        activation_function: callable = F.relu, 
        use_layer_norm: bool = True,
        edge_norm: bool = True,
        device: str = "cuda:0"
    ):
        """
        Initialize the Heterogeneous CompGCN model.
        
        Args:
            in_channels_dict: Dictionary mapping node types to their input feature dimensions
                             (None means learnable embeddings)
            mlp_out_emb_size: Output dimension of node type specific MLPs
            conv_hidden_channels: Dictionary mapping layer names to their hidden dimensions
            num_nodes_per_type: Dictionary mapping node types to their counts
            num_entities: Total number of entities
            num_relations: Number of relation types
            conv_num_layers: Number of CompGCN convolutional layers
            opn: Composition operation ('mult', 'sub', or 'corr')
            dropout: Dropout rate
            activation_function: Activation function to use between layers
            use_layer_norm: Whether to use layer normalization
            edge_norm: Whether to use edge normalization
            device: Device to run the model on
        """
        super().__init__()

        self.mlp_out = mlp_out_emb_size
        self.device = device
        self.num_entities = num_entities
        self.num_nodes_per_type = num_nodes_per_type
        self.activation_function = activation_function
        self.layers_num = conv_num_layers
        self.use_layer_norm = use_layer_norm

        # Node type embedding projection
        self.mlp_dict = nn.ModuleDict()
        self.node_type_offset = {}
        offset = 0
        for node_type, in_channels in in_channels_dict.items():
            self.node_type_offset[node_type] = offset
            if in_channels is None:
                self.mlp_dict[node_type] = nn.Embedding(num_nodes_per_type[node_type], mlp_out_emb_size)
            else:
                self.mlp_dict[node_type] = nn.Sequential(
                    nn.Linear(in_channels, mlp_out_emb_size),
                    nn.Dropout(dropout)
                )
            offset += num_nodes_per_type[node_type]

        # Layer-wise relation embeddings (including inverses)
        self.relation_embeddings_per_layer = nn.ParameterList()
        for idx in range(conv_num_layers):
            in_dim = conv_hidden_channels[f'layer_{idx-1}'] if idx > 0 else mlp_out_emb_size
            rel_emb = nn.Parameter(torch.Tensor(2 * num_relations, in_dim))
            if opn == 'corr':
                nn.init.eye_(rel_emb)
            else:
                nn.init.xavier_uniform_(rel_emb)
            self.relation_embeddings_per_layer.append(rel_emb)

        # CompGCN layers
        self.conv_layers = nn.ModuleList()
        
        # Layer normalization
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        for idx in range(conv_num_layers):
            in_channels = conv_hidden_channels[f'layer_{idx-1}'] if idx > 0 else mlp_out_emb_size
            out_channels = conv_hidden_channels[f'layer_{idx}']
            
            self.conv_layers.append(
                CompGCNConv(
                    in_channels,
                    out_channels,
                    num_relations,
                    comp_fn=opn,
                    dropout=dropout,
                    edge_norm=edge_norm
                )
            )
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_channels))

    def forward(self, x_dict: Dict[str, Optional[torch.Tensor]], edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the heterogeneous CompGCN model.
        
        Args:
            x_dict: Dictionary mapping node types to their input features 
                  (None means use embeddings)
            edge_index: Edge information [num_edges, 3] with (src, rel_type, dst)
            
        Returns:
            Updated node embeddings [num_entities, output_dim]
        """
        # Initialize node embeddings
        x_all = torch.zeros(self.num_entities, self.mlp_out, device=self.device)
        for node_type, features in x_dict.items():
            offset = self.node_type_offset[node_type]
            if features is None:
                # Use learnable embeddings
                emb = self.mlp_dict[node_type](
                    torch.arange(0, self.num_nodes_per_type[node_type], device=self.device)
                )
            else:
                # Use provided features
                emb = self.mlp_dict[node_type](features)
            x_all[offset:offset + self.num_nodes_per_type[node_type]] = emb

        # Pass through GCN layers
        x = x_all
        for layer_idx in range(self.layers_num):
            # Transform relation embeddings using the relation-specific weight matrix
            # rel_emb = self.conv_layers[layer_idx].w_rel(self.relation_embeddings_per_layer[layer_idx])
            rel_emb = self.relation_embeddings_per_layer[layer_idx]
            
            # Apply CompGCN convolution
            x = self.conv_layers[layer_idx](x, rel_emb, edge_index)
            
            # Apply activation for all but the last layer
            if layer_idx < self.layers_num - 1:
                x = self.activation_function(x)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[layer_idx](x)
                

        # Store final relation embeddings for scoring
        self.final_rel_emb = self.conv_layers[-1].w_rel(self.relation_embeddings_per_layer[-1])
        return x

    def distmult(self, embedding: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        """
        Score triplets using DistMult scoring function.
        
        Args:
            embedding: Entity embeddings
            triplets: Triplets to score [batch_size, 3] with (subject, relation, object)
            
        Returns:
            Scores for each triplet [batch_size]
        """
        s = embedding[triplets[:, 0]]
        r = self.final_rel_emb[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def complex(self, embedding: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        """
        Score triplets using ComplEx scoring function.
        Assuming the embedding has been structured as [real; imag]
        
        Args:
            embedding: Entity embeddings with real and imaginary parts concatenated
            triplets: Triplets to score [batch_size, 3] with (subject, relation, object)
            
        Returns:
            Scores for each triplet [batch_size]
        """
        dim = embedding.size(1) // 2
        
        # Split into real and imaginary parts
        s_re, s_im = embedding[triplets[:, 0], :dim], embedding[triplets[:, 0], dim:]
        r_re, r_im = self.final_rel_emb[triplets[:, 1], :dim], self.final_rel_emb[triplets[:, 1], dim:]
        o_re, o_im = embedding[triplets[:, 2], :dim], embedding[triplets[:, 2], dim:]
        
        # ComplEx scoring function
        return torch.sum(
            (s_re * r_re * o_re) + 
            (s_re * r_im * o_im) + 
            (s_im * r_re * o_im) - 
            (s_im * r_im * o_re), 
            dim=1
        )

    def score_loss(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate binary cross entropy loss for the scores.
        
        Args:
            scores: Predicted scores
            target: Target labels (0 or 1)
            
        Returns:
            BCE loss
        """
        return F.binary_cross_entropy_with_logits(scores, target)

    def reg_loss(self, embedding: torch.Tensor, triplets: torch.Tensor, lambda_reg: float = 0.01) -> torch.Tensor:
        """
        Calculate L2 regularization loss for entity and relation embeddings.
        
        Args:
            embedding: Entity embeddings
            triplets: Triplets used for scoring
            lambda_reg: Regularization strength
            
        Returns:
            Regularization loss
        """
        s_index, p_index, o_index = triplets.t()
        s = embedding[s_index]
        p = self.final_rel_emb[p_index]
        o = embedding[o_index]
        
        # L2 regularization
        reg = lambda_reg * (
            torch.mean(s.pow(2)) + 
            torch.mean(p.pow(2)) + 
            torch.mean(o.pow(2))
        )
        
        return reg
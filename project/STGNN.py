import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class STGNN(nn.Module):
    """
    Spatio-Temporal GNN for Estonian energy resilience.
    
    Spatial:  Residual GATv2 with Physical Edge Weights
    Temporal: GRU with Global Context
    """
    def __init__(self, in_dim, hidden_dim, num_quantiles=3, edge_dim=1):
        super().__init__()
        # Added edge_dim to support physical cable capacities
        self.gat1 = gnn.GATv2Conv(in_dim, hidden_dim, heads=4,
                                   concat=False, dropout=0.2, edge_dim=edge_dim)
        self.gat2 = gnn.GATv2Conv(hidden_dim, hidden_dim, heads=4,
                                   concat=False, dropout=0.2, edge_dim=edge_dim)
        
        # Residual projection (Skip connection) to match in_dim to hidden_dim
        self.residual_proj = nn.Linear(in_dim, hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim,
                            num_layers=2, batch_first=True, dropout=0.3)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_quantiles),
        )

    def forward(self, x_seq, edge_index, edge_attr=None):
        B, T, N, F = x_seq.shape

        x_flat = x_seq.reshape(B * T * N, F)
        offsets = torch.arange(B * T, device=x_seq.device).view(-1, 1, 1) * N
        batch_edges = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)
        
        # Expand edge_attr for the vectorized batch if provided
        batch_edge_attr = None
        if edge_attr is not None:
            # edge_attr shape: (E, D) -> repeat for all batches and timesteps
            batch_edge_attr = edge_attr.repeat(B * T, 1)

        # Layer 1 with Residual
        h = torch.relu(self.gat1(x_flat, batch_edges, edge_weight=batch_edge_attr))
        h = h + self.residual_proj(x_flat)
        
        # Layer 2 with Residual
        h_prev = h
        h = torch.relu(self.gat1(x_flat, batch_edges,edge_attr=batch_edge_attr))
        h = h + h_prev

        h_reshaped = h.view(B, T, N, -1)
        combined_context = torch.mean(h_reshaped, dim=2) 

        _, h_n = self.gru(combined_context)
        return self.decoder(h_n[-1])
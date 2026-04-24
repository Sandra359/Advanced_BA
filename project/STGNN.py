import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class STGNN(nn.Module):
    """
    Spatio-Temporal GNN for Estonian energy resilience.
    Spatial:  GATv2 with residual connection
    Temporal: GRU over EE node embeddings
    Output:   3 quantiles (P10, P50, P90)
    """
    def __init__(self, in_dim, hidden_dim, num_quantiles=3):
        super().__init__()
        self.gat1 = gnn.GATv2Conv(in_dim,     hidden_dim, heads=4,
                                   concat=False, dropout=0.2)
        self.gat2 = gnn.GATv2Conv(hidden_dim, hidden_dim, heads=4,
                                   concat=False, dropout=0.2) #dropout 0.2 is more appropriate for small datasets to prevent overfitting, can be tuned as a hyperparameter

        # Residual projection: maps input dim → hidden dim for skip connection
        self.residual_proj = nn.Linear(in_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim,
                           num_layers=2, batch_first=True, dropout=0.2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_quantiles),
        )

    def forward(self, x_seq, edge_index):
        B, T, N, F = x_seq.shape

        # Vectorized: process all B*T graphs at once
        x_flat      = x_seq.reshape(B * T * N, F)
        offsets     = torch.arange(B * T, device=x_seq.device).view(-1, 1, 1) * N
        batch_edges = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)

        # GAT layer 1 + residual skip connection
        h = torch.relu(self.gat1(x_flat, batch_edges))
        h = h + self.residual_proj(x_flat)  # skip: project input to hidden dim

        # GAT layer 2 + residual skip connection
        h_prev = h
        h = torch.relu(self.gat2(h, batch_edges))  # gat2 not gat1 here
        h = h + h_prev                              # skip: same dim now

        # Extract EE node (node 0) embeddings → temporal sequence
        ee_seq = h.view(B, T, N, -1)[:, :, 0, :]  # (B, T, hidden_dim)

        # GRU over time
        _, h_n = self.gru(ee_seq)
        return self.decoder(h_n[-1])  # (B, 3)
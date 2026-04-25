import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class STGNN(nn.Module):
    """
    Spatio-Temporal GNN for Estonian energy resilience.
    Spatial:  GATv2 with residual connection
    Temporal: GRU over EE node embeddings
    Output:   3 quantiles (P10, P50, P90)

    Changes vs v1:
    - BatchNorm1d → LayerNorm in decoder: BatchNorm uses running stats
      estimated on training data, which breaks under distribution shift
      (e.g. Jan 2026 test set). LayerNorm normalizes per sample and is
      therefore more robust at test time.
    - hidden_dim exposed properly; recommend 32–64 (was 16, too small).
    - Dropout comment corrected: 0.3 is used consistently.
    """
    def __init__(self, in_dim, hidden_dim, num_quantiles=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gat1 = gnn.GATv2Conv(in_dim,     hidden_dim, heads=4,
                                   concat=False, dropout=0.3)
        self.gat2 = gnn.GATv2Conv(hidden_dim, hidden_dim, heads=4,
                                   concat=False, dropout=0.3)

        # Residual projection: maps input dim → hidden dim for skip connection
        self.residual_proj = nn.Linear(in_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim,
                           num_layers=2, batch_first=True, dropout=0.3)

        # FIX: LayerNorm instead of BatchNorm1d.
        # BatchNorm1d accumulates running mean/var from training data and
        # applies those at eval time — this breaks when test distribution
        # differs from training (temporal split). LayerNorm normalises
        # per sample and has no running statistics, so it generalises
        # correctly under distribution shift.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),   # ← was BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_quantiles),
        )

    def forward(self, x_seq, edge_index):
        B, T, N, F = x_seq.shape

        # Vectorized: process all B*T graphs at once
        x_flat      = x_seq.reshape(B * T * N, F)
        offsets     = torch.arange(B * T, device=x_seq.device).view(-1, 1, 1) * N
        batch_edges = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)

        # GAT layer 1 + residual skip connection
        h = torch.relu(self.gat1(x_flat, batch_edges))
        h = h + self.residual_proj(x_flat)

        # GAT layer 2 + residual skip connection
        h_prev = h
        h = torch.relu(self.gat2(h, batch_edges))
        h = h + h_prev

        # Extract EE node (node 0) embeddings → temporal sequence
        ee_seq = h.view(B, T, N, -1)[:, :, 0, :]  # (B, T, hidden_dim)

        # GRU over time
        _, h_n = self.gru(ee_seq)
        return self.decoder(h_n[-1])  # (B, 3)
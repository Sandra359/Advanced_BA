import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class STGNN(nn.Module):
    """
    Spatio-Temporal GNN for Estonian energy resilience.
    Spatial:  GATv2 with residual connection
    Temporal: GRU over EE node embeddings + neighbour mean
    Output:   3 quantiles (P10, P50, P90) — monotonicity enforced via sort

    """

    def __init__(self, in_dim, hidden_dim, num_quantiles=3, seq_len=168, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Spatial layers ---
        # edge_dim=1 initialises lin_edge so edge_attr (flow magnitudes) can
        # be passed at forward time. Without this GATv2Conv raises AssertionError.
        self.gat1 = gnn.GATv2Conv(in_dim, hidden_dim, heads=4,
                                   concat=False, dropout=dropout, edge_dim=1)
        self.gat2 = gnn.GATv2Conv(hidden_dim, hidden_dim, heads=4,
                                   concat=False, dropout=dropout, edge_dim=1)

        # Residual projection: maps input dim → hidden dim for skip connection
        self.residual_proj = nn.Linear(in_dim, hidden_dim)

        # [4] Positional embedding over the sequence length
        # Gives the GRU explicit awareness of which timestep it is processing.
        # seq_len must match SEQ_LEN used when building X_list in Supply_wind.py.
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)

        # [2] GRU input is now hidden_dim * 2:
        #     hidden_dim  from EE node embedding
        #     hidden_dim  from mean of neighbour (FI, LV, LT) embeddings
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim,
                           num_layers=2, batch_first=True, dropout=dropout)

        # [1] Explicit dropout after GRU final hidden state.
        # PyTorch GRU dropout=0.3 only applies between the two GRU layers,
        # NOT after the last layer. This Dropout fills that gap.
        self.gru_dropout = nn.Dropout(dropout)

        # Decoder: LayerNorm (not BatchNorm) for robustness under distribution shift
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_quantiles),
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        B, T, N, F = x_seq.shape

        # --- Spatial pass: process all B*T graph snapshots in parallel ---
        x_flat = x_seq.reshape(B * T * N, F)
        offsets = torch.arange(B * T, device=x_seq.device).view(-1, 1, 1) * N
        batch_edges = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)

        # [5] Tile edge_weight across all B*T graphs so it matches batch_edges.
        # edge_weight shape: (num_edges,) → (B*T*num_edges,)
        # If no edge_weight is provided, GAT uses uniform attention priors.
        batch_weight = None
        if edge_weight is not None:
            batch_weight = edge_weight.repeat(B * T).unsqueeze(-1)  # (B*T*E, 1)

        # GAT layer 1 + residual skip from input
        h = torch.relu(self.gat1(x_flat, batch_edges, edge_attr=batch_weight))
        h = h + self.residual_proj(x_flat)

        # GAT layer 2 + residual skip from previous hidden
        h_prev = h
        h = torch.relu(self.gat2(h, batch_edges, edge_attr=batch_weight))
        h = h + h_prev

        # Reshape back to (B, T, N, hidden_dim)
        h = h.view(B, T, N, -1)

        # --- [2] Build GRU input: EE embedding + mean of neighbour embeddings ---
        # EE is node 0; neighbours are nodes 1 (FI), 2 (LV), 3 (LT)
        ee_seq    = h[:, :, 0, :]           # (B, T, hidden_dim)
        neigh_features = h[:, :, 1:, :] # (B, T, 3, hidden_dim)
        attn_weights = torch.softmax(torch.matmul(ee_seq.unsqueeze(2), neigh_features.transpose(-1, -2)), dim=-1)
        
        neigh_seq = torch.matmul(attn_weights, neigh_features).squeeze(2)

        # Concatenate along feature dim → (B, T, hidden_dim * 2)
        gru_input = torch.cat([ee_seq, neigh_seq], dim=-1)

        # --- [4] Add positional embeddings to EE sequence ---
        # positions: (T,) → embed to (T, hidden_dim) → broadcast over batch
        # Added only to the EE component (first hidden_dim slice) so the
        # positional signal is on the primary node, not the neighbour mean.
        positions = torch.arange(T, device=x_seq.device)
        pos       = self.pos_emb(positions).unsqueeze(0)  # (1, T, hidden_dim)
        gru_input = torch.cat([
            gru_input[:, :, :self.hidden_dim] + pos,   # EE + positional
            gru_input[:, :, self.hidden_dim:],          # neighbour mean unchanged
        ], dim=-1)

        # --- Temporal pass: GRU over the sequence ---
        _, h_n = self.gru(gru_input)

        # [1] Dropout on final hidden state before decoder
        out = self.decoder(self.gru_dropout(h_n[-1]))

        # [3] Enforce quantile monotonicity: P10 ≤ P50 ≤ P90
        # torch.sort is differentiable, so gradients flow through correctly.
        return torch.sort(out, dim=-1).values  # (B, 3)
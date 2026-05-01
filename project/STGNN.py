from graphviz import view
from distutils.command.clean import clean
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import graphviz


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
    
    
dot_code = """
// Spatio-Temporal GNN (STGNN) for Energy Resilience Diagram
// Render using Graphviz or a compatible viewer
digraph STGNN_Architecture {
    rankdir=TD; // Top to Down flow
    node [fontname="Helvetica,Arial,sans-serif", shape=box, style=filled, color="#E1E1E1", fontsize=10];
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=9];

    // --- Graph Styles ---
    // Module Boxes
    subgraph cluster_style_layers { graph[style=invis]; node [fillcolor="#cfe2f3", color="#6fa8dc"]; } // Light Blue for Layers
    subgraph cluster_style_data { graph[style=invis]; node [fillcolor="#ffffff", color="#aaaaaa", shape=none, style=none]; } // White/Plain for Data
    subgraph cluster_style_op { graph[style=invis]; node [fillcolor="#fff2cc", color="#ffd966", shape=ellipse]; } // Yellow/Ellipse for Ops

    //=========================================
    // 1. INPUTS
    //=========================================
    subgraph cluster_0 {
        label = "1. Input Sequence (Graph Time Series)";
        style="filled,dashed"; color="#f3f3f3"; fontname="Helvetica-Bold"; fontsize=12;

        x_seq [label=<<b>Input Sequence x_seq</b><br/>(B, T, N, F)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];
        edge_index [label=<<b>edge_index</b><br/>(2, num_edges)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];
        edge_weight [label=<<b>edge_weight</b><br/>(num_edges,)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];
    }

    //=========================================
    // 2. SPATIO-TEMPORAL GNN BLOCK
    //=========================================
    subgraph cluster_1 {
        label = "2. Spatial Pass (Process Snapshots in Parallel)";
        style="filled"; color="#f1f1f1"; fontname="Helvetica-Bold"; fontsize=12;

        node [fillcolor="#cfe2f3", color="#6fa8dc"]; // Reset for layers
        
        // Flattening & Prepping
        reshape_1 [label=<<b>Flatten (B*T Snapshots)</b><br/>x_seq → x_flat<br/>N=4 (EE, FI, LV, LT)> shape=parallelogram];
        t_w_1 [label="Tile Edge Weights" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];

        // GAT Layer 1
        subgraph cluster_1a {
            label = "GAT Block 1"; style=filled; color="#cfe2f3"; fontname="Helvetica-Oblique";
            gat1 [label=<<b>GATv2Conv (gat1)</b><br/>In: F, Out: H<br/>Uses flow edge weights>];
            relu1 [label="ReLU" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];
            lin_res [label=<<b>Residual Projection (lin)</b><br/>Map F → H>];
            add1 [label="+" shape=circle fillcolor="#f3f3f3" color="#aaaaaa"];
        }

        // GAT Layer 2
        subgraph cluster_1b {
            label = "GAT Block 2"; style=filled; color="#cfe2f3"; fontname="Helvetica-Oblique";
            gat2 [label=<<b>GATv2Conv (gat2)</b><br/>In: H, Out: H<br/>Uses flow edge weights>];
            relu2 [label="ReLU" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];
            add2 [label="+" shape=circle fillcolor="#f3f3f3" color="#aaaaaa"];
        }

        // Output shaping
        reshape_2 [label=<<b>Reshape to Sequence</b><br/>→ (B, T, N, H)> shape=parallelogram];

        // Logical Flow inside Spatial Block
        x_seq -> reshape_1 [label="(B, T, N, F)"];
        reshape_1 -> lin_res;
        
        reshape_1 -> gat1 [label="(B*T*N, F)"];
        edge_index -> gat1;
        edge_weight -> t_w_1;
        t_w_1 -> gat1 [label="(B*T*E, 1)"];
        gat1 -> relu1;
        relu1 -> add1;
        lin_res -> add1;

        add1 -> gat2 [label="(B*T*N, H)"];
        edge_index -> gat2;
        t_w_1 -> gat2;
        gat2 -> relu2;
        add1 -> add2; // Residual connect previous hidden
        relu2 -> add2;
        add2 -> reshape_2 [label="(B*T*N, H)"];
    }

    //=========================================
    // 3. SPATIAL AGGREGATION & TEMPORAL PREP
    //=========================================
    subgraph cluster_2 {
        label = "3. Neighbor & Positional Context Prep";
        style="filled,dashed"; color="#fffaf0"; fontname="Helvetica-Bold"; fontsize=12;

        node [fillcolor="#ffffff", color="#aaaaaa", shape=none, style=none];

        // Slicing Nodes
        split_node [label="Split by Node Index" shape=diamond fillcolor="#fff2cc" color="#ffd966"];
        ee_features [label=<<b>EE Node features</b><br/>node 0, T steps<br/>(B, T, H)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];
        neigh_features [label=<<b>Neighbor features</b><br/>nodes [1:4] (FI,LV,LT), T steps<br/>(B, T, 3, H)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];

        // Weighted Aggregation
        attn_calc [label="Attention: MatMul & Softmax" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];
        apply_attn [label="Weighted Sum" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];
        neigh_agg [label=<<b>Aggregated Neighbors</b><br/>(B, T, H)> fillcolor="#eeeeee" shape=parallelogram style=filled];

        // Concatenation and Positional Embedding
        pos_emb [label=<<b>Positional Embedding (pos_emb)</b><br/>Lookup per timestep 0..T> fillcolor="#cfe2f3" color="#6fa8dc"];
        add_pos [label="+ (Add pos signal)" shape=circle fillcolor="#f3f3f3" color="#aaaaaa"];
        concat_struc [label=<<b>Concatenate Structure</b><br/>[EE_context, Neigh_context]> shape=parallelogram fillcolor="#eeeeee" style=filled];
        gru_input [label=<<b>GRU Input Sequence</b><br/>(B, T, 2*H)> fillcolor="#e6b8af" color="#dd7e6b" shape=parallelogram style=filled];

        // Connect flow
        reshape_2 -> split_node [label="(B, T, N, H)"];
        split_node -> ee_features;
        split_node -> neigh_features;

        ee_features -> attn_calc;
        neigh_features -> attn_calc [label="(transpose)"];
        attn_calc -> apply_attn [label="Weights (B,T,1,3)"];
        neigh_features -> apply_attn;
        apply_attn -> neigh_agg;

        ee_features -> add_pos;
        pos_emb -> add_pos;

        add_pos -> concat_struc [label="EE Context (B,T,H)"];
        neigh_agg -> concat_struc [label="Neigh Context (B,T,H)"];
        concat_struc -> gru_input;
    }

    //=========================================
    // 4. TEMPORAL ENCODING & DECODING
    //=========================================
    subgraph cluster_3 {
        label = "4. Temporal Gating & Quantile Decoding";
        style="filled"; color="#f1f1f1"; fontname="Helvetica-Bold"; fontsize=12;

        node [fillcolor="#d9ead3", color="#93c47d"]; // Green for Temporal/Output

        gru [label=<<b>GRU Layer (gru)</b><br/>2 layers, dropout=0.3<br/>Input: 2*H, Out: H>];
        last_state [label=<<b>Last Hidden State (h_n[-1])</b><br/>(B, H)> fillcolor="#eeeeee" shape=parallelogram style=filled color="#aaaaaa"];
        
        gru_drop [label=<<b>GRU Final Dropout (gru_dropout)</b><br/>rate=0.3>];

        // Decoder Stack
        subgraph cluster_3a {
            label = "Decoder (Sequential)"; style=filled; color="#d9ead3"; fontname="Helvetica-Oblique";
            dec_lin1 [label=<<b>Linear</b><br/>H → 2*H>];
            dec_ln [label=<<b>LayerNorm</b><br/>(not BatchNorm!)>];
            dec_relu [label="ReLU" shape=ellipse fillcolor="#fff2cc" color="#ffd966"];
            dec_drop [label=<<b>Dropout</b><br/>rate=0.3>];
            dec_lin2 [label=<<b>Linear</b><br/>2*H → 3 (Quantiles)>];
        }

        raw_out [label=<<b>Raw Predicted Quantiles</b><br/>(B, 3)> fillcolor="#eeeeee" shape=parallelogram style=filled color="#aaaaaa"];
        sort [label=<<b>torch.sort</b><br/>(Enforce P10 ≤ P50 ≤ P90)<br/>Differentiable> shape=ellipse fillcolor="#fff2cc" color="#ffd966"];

        // Output Parallelogram
        final_output [label=<<b>Final Prediction</b><br/>Ordered Quantiles<br/>(B, 3)> fillcolor="#fce5cd" color="#e06666" shape=parallelogram style="filled,bold"];

        // Connect flow
        gru_input -> gru;
        gru -> last_state;
        last_state -> gru_drop;
        gru_drop -> dec_lin1;
        dec_lin1 -> dec_ln;
        dec_ln -> dec_relu;
        dec_relu -> dec_drop;
        dec_drop -> dec_lin2;
        dec_lin2 -> raw_out;
        raw_out -> sort;
        sort -> final_output;
    }
}


"""


dot_code2 = """
digraph STGNN_Short {
    rankdir=TD;
    node [fontname="Arial", shape=box, style=filled, fontsize=10, color="#666666"];
    edge [arrowsize=0.7, color="#888888"];

    // Global Styles
    node [fillcolor="#e6b8af"] x_seq, edge_idx, edge_wt; // Inputs
    node [fillcolor="#cfe2f3"] GAT1, GAT2, res_proj;    // Spatial
    node [fillcolor="#fff2cc"] attn, pos_emb, sort;      // Operations
    node [fillcolor="#d9ead3"] GRU, decoder;            // Temporal/Output

    // 1. Spatial Pass (Flattened)
    {x_seq, edge_idx, edge_wt} -> GAT1;
    x_seq -> res_proj -> add1;
    GAT1 -> add1 -> GAT2 -> add2;
    add1 -> add2 [label="res"];

    // 2. Aggregation & Positional Logic
    add2 -> {ee_feat, neigh_feat} [label="split"];
    {ee_feat, neigh_feat} -> attn -> neigh_agg;
    ee_feat -> pos_add;
    pos_emb -> pos_add -> concat;
    neigh_agg -> concat;

    // 3. Temporal & Decode
    concat -> GRU -> gru_drop -> decoder -> sort -> final_pred;

    // Layout Groupings
    subgraph cluster_spatial { label="Spatial"; GAT1; GAT2; res_proj; }
    subgraph cluster_temporal { label="Temporal"; GRU; decoder; sort; }
}"""
    
    
    
if __name__ == "__main__":

    
    model = STGNN(in_dim=16, hidden_dim=32)
    print(model)

    # Visualize the architecture with graphviz
    graph = graphviz.Source(dot_code)
    graph.render(format="png", outfile="../figures/stgnn_architecture.png", cleanup=True, view=True)
    
    png_bytes = graph.pipe(format="png")

    with open("../figures/stgnn_architecture.png", "wb") as f:
        f.write(png_bytes)
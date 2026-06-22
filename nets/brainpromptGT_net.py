import math
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.mlp_readout_layer import MLPReadout, simpleGCNLayer
from layers.gcn_layer import simpleGCNLayer
from layers.attention_layer import LMPE


# ── Graph Transformer Layer ───────────────────────────────────────────────────
class GTLayer(nn.Module):
    """
    Graph Transformer layer with multi-head attention.
    Replaces GCNLayer for ROI-level message passing.
    Each ROI attends to its neighbors weighted by learned attention scores.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.0,
                 batch_norm=True, residual=True):
        super().__init__()
        self.num_heads   = num_heads
        self.out_dim     = out_dim
        self.head_dim    = out_dim // num_heads
        self.residual    = residual
        self.batch_norm  = batch_norm

        assert out_dim % num_heads == 0, \
            f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"

        # Q, K, V projections
        self.Q = nn.Linear(in_dim,  out_dim, bias=False)
        self.K = nn.Linear(in_dim,  out_dim, bias=False)
        self.V = nn.Linear(in_dim,  out_dim, bias=False)

        # Output projection
        self.O = nn.Linear(out_dim, out_dim, bias=False)

        # Residual projection if dims differ
        self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) \
            if in_dim != out_dim else nn.Identity()

        self.dropout    = nn.Dropout(dropout)
        self.batch_norm_h = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

    def forward(self, g, h, e=None):
        """
        g : DGL graph
        h : node features  (N, in_dim)
        e : edge features  (E, edge_dim)  — kept for API compatibility, not used
        """
        with g.local_scope():
            B, H, D = h.shape[0], self.num_heads, self.head_dim

            # Project to Q, K, V and reshape to (N, heads, head_dim)
            Q_h = self.Q(h).view(B, H, D)   # (N, H, D)
            K_h = self.K(h).view(B, H, D)
            V_h = self.V(h).view(B, H, D)

            g.ndata['Q'] = Q_h
            g.ndata['K'] = K_h
            g.ndata['V'] = V_h

            # ── Attention: e_ij = (Q_i · K_j) / sqrt(d) ─────────────────────
            g.apply_edges(fn.u_dot_v('K', 'Q', 'attn'))   # (E, H, 1)
            attn = g.edata['attn'] / math.sqrt(D)          # scale
            attn = self.dropout(
                dgl.ops.edge_softmax(g, attn))              # softmax per node

            # ── Aggregate: h_i = sum_j( attn_ij * V_j ) ─────────────────────
            g.edata['attn'] = attn
            g.update_all(
                fn.u_mul_e('V', 'attn', 'm'),
                fn.sum('m', 'h_new')
            )
            h_new = g.ndata['h_new'].view(B, H * D)        # (N, out_dim)
            h_new = self.O(h_new)

            # ── Residual + BN ─────────────────────────────────────────────────
            if self.residual:
                h_new = h_new + self.residual_proj(h)
            h_new = self.batch_norm_h(h_new)
            h_new = F.relu(h_new)

        return h_new, e   # return e unchanged (API compatibility)


# ── BrainPromptG with Graph Transformer ──────────────────────────────────────
class BrainPromptGTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.name = 'BrainPromptGT'

        in_dim          = net_params['in_dim']
        edge_dim        = net_params['edge_dim']
        hidden_dim      = net_params['hidden_dim']
        out_dim         = net_params['out_dim']
        n_classes       = net_params['n_classes']
        dropout         = net_params['dropout']
        n_layers        = net_params['L']
        num_heads       = net_params.get('n_heads', 4)   # ← new param
        self.readout    = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual   = net_params['residual']
        self.e_feat     = net_params['edge_feat']
        self.node_num   = net_params['node_num']
        self.label_embs = net_params['label_embs']
        self.lambda1    = net_params['lambda1']
        self.lm_dim     = 384

        # ── Input projections ─────────────────────────────────────────────────
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.dropout     = nn.Dropout(p=dropout)

        # ── Graph Transformer layers (replaces GCNLayer) ──────────────────────
        self.gt_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.gt_layers.append(
                GTLayer(hidden_dim, hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_norm=self.batch_norm,
                        residual=self.residual)
            )
        # Final layer → out_dim
        self.gt_layers.append(
            GTLayer(hidden_dim, out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_norm=self.batch_norm,
                    residual=self.residual)
        )

        # ── LLM + Population graph components ────────────────────────────────
        self.MLP_layer      = MLPReadout(out_dim, n_classes)
        self.linear_layer1  = nn.Linear(self.lm_dim, out_dim)
        self.pos_emb        = LMPE(emb_dim=self.lm_dim, d_model=out_dim,
                                   node_num=self.node_num, dropout=dropout)
        self.linear_layer2  = nn.Linear(out_dim, out_dim)
        self.label_transform = nn.Linear(self.lm_dim, out_dim)
        self.fused_repr     = None
        self.global_gcs     = simpleGCNLayer(hidden_dim, F.relu, 0.0,
                                             batch_norm=False, residual=True)

    def forward(self, g, h, e, batch_llms):
        device = h.device

        # ── Node embedding ────────────────────────────────────────────────────
        h = self.embedding_h(h)

        # ── ROI positional encoding (LLM-based) ──────────────────────────────
        hidden_dim = h.shape[-1]
        h = self.pos_emb(
            h.reshape(-1, self.node_num, hidden_dim)
        ).reshape(-1, hidden_dim)

        # ── Graph Transformer layers (ROI-level message passing) ──────────────
        for gt_layer in self.gt_layers:
            h, e = gt_layer(g, h, e)

        # ── Graph readout ─────────────────────────────────────────────────────
        g.ndata['h'] = h
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        # ── Subject-level LLM prompt fusion ───────────────────────────────────
        llm       = batch_llms.squeeze(1)
        meta_repr = self.linear_layer1(llm)

        # ── Population graph (subject similarity) ─────────────────────────────
        hg_sim        = self.sim(hg)
        llm_sim       = self.sim(meta_repr)
        binary_matrix = self.dropout(((hg_sim * llm_sim) > 0.8).float())
        fused_repr    = self.global_gcs(hg, binary_matrix)
        fused_repr    = fused_repr + hg + meta_repr

        self.fused_repr  = fused_repr
        self.label_reprs = self.label_transform(self.label_embs.to(device))

        scores = self.MLP_layer(fused_repr)
        return scores

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    def compute_label_loss(self, label, weight=None):
        text_features  = self.label_reprs
        graph_features = self.fused_repr / self.fused_repr.norm(dim=1, keepdim=True)
        text_features  = text_features   / text_features.norm(dim=1, keepdim=True)
        logits         = graph_features @ text_features.t()
        return F.cross_entropy(logits, label, weight=weight)

    def sim(self, matrix):
        x_norm     = F.normalize(matrix, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        return torch.sigmoid(sim_matrix)
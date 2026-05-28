import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from AbMEGD.modules.common.layers import mask_zero, LayerNorm
from AbMEGD.modules.common.geometry import global_to_local, local_to_global, normalize_vector


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: (N, L_i, L_j, num_heads)
        mask:   (N, L)
    Returns:
        alpha:  (N, L_i, L_j, num_heads)
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x: (..., n_heads * n_ch)
    Returns:
        (..., n_heads, n_ch)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):
    """
    Geometry-aware attention block with atom-aware conditioning.

    Current full model:
      - atom information is kept at atom resolution until query-conditioned late pooling
      - direct four-route competition:
            node + pair + spatial + atom
      - pair-aware gate is applied only on the atom route aggregation

    Supported ablation_mode:
      - None            : full current model
      - "no_pair_gate"  : current atom route without pair-aware gate
                           (atom route remains query-conditioned late pooling;
                            only gate is removed)
    """

    def __init__(
        self,
        node_feat_dim,
        pair_feat_dim,
        atom_feat_dim,
        vec_dim,
        vec_feat_dim,
        value_dim=32,
        query_key_dim=32,
        num_query_points=8,
        num_value_points=8,
        num_heads=12,
        bias=False,
        max_num_atoms=5,
        ablation_mode=None,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.atom_feat_dim = atom_feat_dim
        self.vec_dim = vec_dim
        self.vec_feat_dim = vec_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads
        self.max_num_atoms = max_num_atoms
        self.ablation_mode = ablation_mode

        # ------------------------------------------------------------
        # Standard residue-level routes
        # ------------------------------------------------------------
        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(
            torch.full(
                [1, 1, 1, self.num_heads],
                fill_value=np.log(np.exp(1.0) - 1.0),
            ),
            requires_grad=True,
        )
        self.proj_query_point = nn.Linear(
            node_feat_dim, num_query_points * num_heads * 3, bias=bias
        )
        self.proj_key_point = nn.Linear(
            node_feat_dim, num_query_points * num_heads * 3, bias=bias
        )
        self.proj_value_point = nn.Linear(
            node_feat_dim, num_value_points * num_heads * 3, bias=bias
        )

        # ------------------------------------------------------------
        # Atom-aware route
        # ------------------------------------------------------------
        self.atom_scalar_proj = nn.Linear(atom_feat_dim, vec_feat_dim, bias=True)
        self.atom_vec_proj = nn.Linear(vec_dim, vec_feat_dim, bias=True)
        self.atom_slot_embed = nn.Embedding(max_num_atoms, vec_feat_dim)

        self.atom_scalar_norm = LayerNorm(vec_feat_dim)
        self.atom_vec_norm = LayerNorm(vec_feat_dim)
        self.atom_hidden_norm_1 = LayerNorm(vec_feat_dim)
        self.atom_hidden_norm_2 = LayerNorm(vec_feat_dim)

        self.atom_transition = nn.Sequential(
            nn.Linear(vec_feat_dim, vec_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(vec_feat_dim * 2, vec_feat_dim),
        )

        # Query-conditioned late pooling
        self.atom_query_norm = LayerNorm(node_feat_dim)
        self.atom_pair_norm = LayerNorm(pair_feat_dim)

        self.proj_atom_query = nn.Linear(node_feat_dim, num_heads * query_key_dim, bias=True)
        self.proj_atom_key = nn.Linear(vec_feat_dim, num_heads * query_key_dim, bias=True)
        self.proj_atom_value = nn.Linear(vec_feat_dim, num_heads * value_dim, bias=True)

        # Scalar route for atom-route logits
        self.proj_atom_logit_value = nn.Linear(vec_feat_dim, num_heads, bias=True)

        # Pair-aware gate over atom route
        self.proj_atom_pair_gate = nn.Linear(pair_feat_dim, num_heads, bias=True)

        # ------------------------------------------------------------
        # Output
        # ------------------------------------------------------------
        in_features = (
            (num_heads * pair_feat_dim)
            + (num_heads * value_dim)
            + (num_heads * num_value_points * (3 + 3 + 1))
            + (num_heads * value_dim)
        )
        self.out_transform = nn.Linear(in_features=in_features, out_features=node_feat_dim)

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.layer_norm_2 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
            nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim),
        )

    # ------------------------------------------------------------------
    # Standard residue-level routes
    # ------------------------------------------------------------------
    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)
        logits_node = (
            query_l.unsqueeze(2) * key_l.unsqueeze(1) * (1 / np.sqrt(self.query_key_dim))
        ).sum(-1)
        return logits_node

    def _pair_logits(self, z):
        return self.proj_pair_bias(z)

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()

        query_points = _heads(
            self.proj_query_point(x),
            self.num_heads * self.num_query_points,
            3,
        )
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.num_heads, -1)

        key_points = _heads(
            self.proj_key_point(x),
            self.num_heads * self.num_query_points,
            3,
        )
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.num_heads, -1)

        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * (
            (-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2
        )
        return logits_spatial

    # ------------------------------------------------------------------
    # Atom-aware route
    # ------------------------------------------------------------------
    def _build_atom_hidden(self, aaa_feat, vec, mask_atoms):
        """
        Build per-atom hidden states from atom.py outputs only.

        Args:
            aaa_feat:   (N*L*A, atom_feat_dim)
            vec:        (N*L*A, 8, 64)
            mask_atoms: (N, L, A)
        Returns:
            atom_hidden: (N, L, A, vec_feat_dim)
        """
        N, L, A = mask_atoms.shape
        if A > self.max_num_atoms:
            raise ValueError(
                f"mask_atoms has A={A}, but max_num_atoms={self.max_num_atoms}. "
                f"Please increase max_num_atoms in GABlock."
            )

        atom_scalar = aaa_feat.reshape(N, L, A, self.atom_feat_dim)
        atom_vec = vec.reshape(N, L, A, self.vec_dim)

        atom_scalar = self.atom_scalar_norm(self.atom_scalar_proj(atom_scalar))
        atom_vec = self.atom_vec_norm(self.atom_vec_proj(atom_vec))

        slot_idx = torch.arange(A, device=aaa_feat.device)
        slot_feat = self.atom_slot_embed(slot_idx).view(1, 1, A, self.vec_feat_dim)

        atom_hidden = atom_scalar + atom_vec + slot_feat
        atom_hidden = mask_zero(mask_atoms.unsqueeze(-1), atom_hidden)
        atom_hidden = self.atom_hidden_norm_1(atom_hidden)

        atom_hidden = atom_hidden + self.atom_transition(atom_hidden)
        atom_hidden = mask_zero(mask_atoms.unsqueeze(-1), atom_hidden)
        atom_hidden = self.atom_hidden_norm_2(atom_hidden)
        return atom_hidden

    def _atom_route(self, x, z, aaa_feat, vec, mask_atoms):
        """
        Query-conditioned late pooling over source-residue atoms.

        Compared with the old implementation:
            atom -> residue mean -> q/k/v
        this implements:
            residue query x_i -> select atoms inside source residue j -> pooled summary m_ij

        Returns:
            logits_atom    : (N, L_i, L_j, H)
            beta           : (N, L_i, L_j, A, H)
            atom_values    : (N, L_j, A, H, V)
            atom_pair_gate : (N, L_i, L_j, H)
        """
        N, L, A = mask_atoms.shape
        source_has_atoms = mask_atoms.any(dim=-1)  # (N, L)

        atom_hidden = self._build_atom_hidden(aaa_feat, vec, mask_atoms)  # (N, L, A, C_atom)

        # Query from target residues
        q = _heads(
            self.proj_atom_query(self.atom_query_norm(x)),
            self.num_heads,
            self.query_key_dim,
        )  # (N, L_i, H, D)

        # Keys/values from source residue atoms
        k = _heads(
            self.proj_atom_key(atom_hidden),
            self.num_heads,
            self.query_key_dim,
        )  # (N, L_j, A, H, D)

        atom_logit_value = self.proj_atom_logit_value(atom_hidden)  # (N, L_j, A, H)

        atom_values = _heads(
            self.proj_atom_value(atom_hidden),
            self.num_heads,
            self.value_dim,
        )  # (N, L_j, A, H, V)

        # Query-conditioned atom selection inside each source residue.
        # beta_{i,j,a,h} = softmax_a(<q_i^h, k_{j,a}^h>)
        beta_logits = torch.einsum(
            "nihd,njahd->nijah",
            q,
            k,
        ) * (1 / np.sqrt(self.query_key_dim))  # (N, L_i, L_j, A, H)

        atom_mask = mask_atoms[:, None, :, :, None]  # (N, 1, L_j, A, 1)
        beta_logits = beta_logits.masked_fill(~atom_mask, -1e5)
        beta = torch.softmax(beta_logits, dim=3)  # over atom slot A
        beta = beta * source_has_atoms[:, None, :, None, None].float()

        # Atom-route residue-pair logits after late pooling
        logits_atom = torch.einsum("nijah,njah->nijh", beta, atom_logit_value)
        logits_atom = logits_atom * source_has_atoms[:, None, :, None].float()

        # Pair-aware gate
        atom_pair_gate = torch.sigmoid(
            self.proj_atom_pair_gate(self.atom_pair_norm(z))
        )  # (N, L_i, L_j, H)

        # Ablation: remove pair-aware gate but keep the rest of the current atom route unchanged
        if self.ablation_mode == "no_pair_gate":
            atom_pair_gate = torch.ones_like(atom_pair_gate)

        return logits_atom, beta, atom_values, atom_pair_gate

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.value_dim)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        return feat_node.reshape(N, L, -1)

    def _atom_aggregation(self, alpha, beta, atom_values, atom_pair_gate):
        """
        alpha:          (N, L_i, L_j, H)
        beta:           (N, L_i, L_j, A, H)
        atom_values:    (N, L_j, A, H, V)
        atom_pair_gate: (N, L_i, L_j, H)

        Computes:
            feat_atom_i^h = sum_j sum_a alpha_{ijh} * gate_{ijh} * beta_{ijah} * v_{jah}
        """
        alpha_atom = alpha * atom_pair_gate  # (N, L_i, L_j, H)
        feat_atom = torch.einsum(
            "nijh,nijah,njahv->nihv",
            alpha_atom,
            beta,
            atom_values,
        )  # (N, L_i, H, V)
        N, L = feat_atom.shape[:2]
        return feat_atom.reshape(N, L, -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(
            self.proj_value_point(x),
            self.num_heads * self.num_value_points,
            3,
        )
        value_points = local_to_global(
            R,
            t,
            value_points.reshape(N, L, self.num_heads, self.num_value_points, 3),
        )
        aggr_points = (
            alpha.reshape(N, L, L, self.num_heads, 1, 1)
            * value_points.unsqueeze(1)
        )
        aggr_points = aggr_points.sum(dim=2)

        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat(
            [
                feat_points.reshape(N, L, -1),
                feat_distance.reshape(N, L, -1),
                feat_direction.reshape(N, L, -1),
            ],
            dim=-1,
        )
        return feat_spatial

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, R, t, x, z, aaa_feat, vec, mask, mask_atoms):
        """
        Args:
            R:          (N, L, 3, 3)
            t:          (N, L, 3)
            x:          (N, L, F_node)
            z:          (N, L, L, F_pair)
            aaa_feat:   (N*L*A, atom_feat_dim)
            vec:        (N*L*A, 8, 64)
            mask:       (N, L)
            mask_atoms: (N, L, A)
        Returns:
            x_updated:  (N, L, F_node)
        """
        # 1) Four-route logits
        logits_atom, beta, atom_values, atom_pair_gate = self._atom_route(
            x, z, aaa_feat, vec, mask_atoms
        )
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)

        # Direct four-route competition (no extra route-wise normalization/scaling)
        logits_sum = logits_node + logits_pair + logits_spatial + logits_atom
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 4), mask)

        # 2) Four-route aggregations
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_atom = self._atom_aggregation(alpha, beta, atom_values, atom_pair_gate)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Direct feature concatenation (no route-wise feature norm/scaling)
        feat_all = self.out_transform(
            torch.cat([feat_p2n, feat_node, feat_spatial, feat_atom], dim=-1)
        )

        # 3) Residual update
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GAEncoder(nn.Module):
    """
    Residue-level encoder with atom-aware GABlocks.

    atom.py remains unchanged. atom_feat must provide:
        aaa_feat   : (N*L*A, atom_feat_dim)
        vec        : (N*L*A, 8, 64)
        mask_atoms : (N, L, A)

    Supported ablation_mode:
      - None
      - "no_pair_gate"
    """

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt=None, ablation_mode=None):
        super(GAEncoder, self).__init__()
        if ga_block_opt is None:
            ga_block_opt = {}

        self.ablation_mode = ablation_mode

        self.blocks = nn.ModuleList(
            [
                GABlock(
                    node_feat_dim=node_feat_dim,
                    pair_feat_dim=pair_feat_dim,
                    atom_feat_dim=int(node_feat_dim * 0.5),
                    vec_dim=512,  # flatten(8, 64)
                    vec_feat_dim=int(node_feat_dim * 0.5),
                    max_num_atoms=5,
                    ablation_mode=self.ablation_mode,
                    **ga_block_opt,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, R, t, atom_feat, res_feat, pair_feat, mask):
        aaa_feat = atom_feat["aaa_feat"]
        vec = atom_feat["vec"]
        mask_atoms = atom_feat["mask_atoms"]

        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, aaa_feat, vec, mask, mask_atoms)

        return res_feat
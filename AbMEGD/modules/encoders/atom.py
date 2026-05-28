import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABCMeta, abstractmethod
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from AbMEGD.modules.common.layers import (
    Distance,
    Sphere,
    ExpNormalSmearing,
    GaussianRBFExpansion,   # 你前面已经加过
    NeighborEmbedding,
    EdgeEmbedding,
    VecLayerNorm,
    CosineCutoff,
    act_class_mapping,
)


logger = logging.getLogger(__name__)


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class GatedEquivariantBlock(nn.Module):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                activation=activation,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                hidden_channels,
                activation=activation,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return x + v.sum() * 0


# ============================================================
# Original ViS-MP branch (kept unchanged)
# ============================================================

class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        )

        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()

        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()

        gain = 0.1
        nn.init.xavier_uniform_(self.q_proj.weight, gain)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight, gain)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight, gain)
        self.s_proj.bias.data.fill_(0)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight, gain)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight, gain)
            nn.init.xavier_uniform_(self.w_trg_proj.weight, gain)

        nn.init.xavier_uniform_(self.vec_proj.weight, gain)
        nn.init.xavier_uniform_(self.dk_proj.weight, gain)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight, gain)
        self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec_j

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


# ============================================================
# True SchNet branch
# ============================================================

class SchNetFilter(nn.Module):
    def __init__(self, num_rbf, num_filters, activation="ssp"):
        super().__init__()
        act_class = act_class_mapping[activation]
        self.net = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            act_class(),
            nn.Linear(num_filters, num_filters),
            act_class(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, edge_attr):
        return self.net(edge_attr)


class SchNetInteractionBlock(nn.Module):
    """
    True SchNet-style scalar-only interaction block:
        w_ij = filter_net(rbf_ij)
        m_ij = W_in x_j \odot w_ij
        m_i  = sum_j m_ij
        v_i  = W_out m_i
        x_i  = x_i + post(v_i)
    """
    def __init__(self, hidden_channels, num_rbf, num_filters=None, activation="ssp"):
        super().__init__()
        if num_filters is None:
            num_filters = hidden_channels

        act_class = act_class_mapping[activation]
        self.filter_net = SchNetFilter(num_rbf, num_filters, activation=activation)
        self.in2f = nn.Linear(hidden_channels, num_filters, bias=False)
        self.f2out = nn.Linear(num_filters, hidden_channels, bias=True)
        self.post = nn.Sequential(
            act_class(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.filter_net.reset_parameters()
        nn.init.xavier_uniform_(self.in2f.weight)
        nn.init.xavier_uniform_(self.f2out.weight)
        self.f2out.bias.data.zero_()
        for m in self.post:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        w_ij = self.filter_net(edge_attr)                # (E, F)
        x_j = self.in2f(x)[src]                          # (E, F)
        m_ij = x_j * w_ij
        m_i = scatter(m_ij, dst, dim=0, dim_size=x.size(0), reduce="add")
        v_i = self.f2out(m_i)
        x = x + self.post(v_i)
        return x


# ============================================================
# True PaiNN branch
# ============================================================

def painn_sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    edge_dist = edge_dist.clamp(min=1e-8)
    n = torch.arange(edge_size, device=edge_dist.device, dtype=edge_dist.dtype) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)


def painn_cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1.0),
        torch.zeros_like(edge_dist),
    )


class PainnMessageBlock(nn.Module):
    """
    Direct PyTorch adaptation of the PaiNN message block:
      - scalar state: (num_atoms, hidden)
      - vector state: (num_atoms, 3, hidden)
    """
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.scalar_message_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.xavier_uniform_(self.filter_layer.weight)
        if self.filter_layer.bias is not None:
            self.filter_layer.bias.data.zero_()

    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        """
        edge:      (E, 2), edge[:,0] is receiver i, edge[:,1] is sender j
        edge_diff: (E, 3), should be R_j - R_i
        """
        edge_dist = edge_dist.clamp(min=1e-8)

        filter_weight = self.filter_layer(
            painn_sinc_expansion(edge_dist, self.edge_size, self.cutoff)
        )
        filter_weight = filter_weight * painn_cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)

        scalar_out = self.scalar_message_mlp(node_scalar)
        filter_out = filter_weight * scalar_out[edge[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim=1,
        )

        message_vector = node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector

        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)

        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)

        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        return new_node_scalar, new_node_vector


class PainnUpdateBlock(nn.Module):
    def __init__(self, node_size: int):
        super().__init__()
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)

        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update_U.weight)
        nn.init.xavier_uniform_(self.update_V.weight)
        if self.update_U.bias is not None:
            self.update_U.bias.data.zero_()
        if self.update_V.bias is not None:
            self.update_V.bias.data.zero_()
        for m in self.update_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(
            mlp_output,
            node_vector.shape[-1],
            dim=1,
        )

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss

        return node_scalar + delta_s, node_vector + delta_v


# ============================================================
# Unified AtomEmbedding
# ============================================================

class AtomEmbedding(nn.Module):
    def __init__(
        self,
        feat_dim,
        max_num_atoms,
        max_aaa_types=6,
        max_num_neighbors=16,
        lmax=2,
        cutoff=5.0,
        num_layers=6,
        num_rbf=32,
        trainable_rbf=False,
        atom_encoder_type="vismp",     # "vismp" | "schnet" | "painn"
        schnet_num_filters=None,
        schnet_activation="ssp",
        painn_edge_size=20,
        painn_pad_to_8=True,
        enable_mode_log=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.hidden_channels = feat_dim
        self.max_aaa_types = max_aaa_types
        self.cutoff = cutoff
        self.max_num_atoms = 5      # keep original behavior for compatibility
        self.max_num_neighbors = max_num_neighbors
        self.atom_encoder_type = atom_encoder_type.lower()

        self.enable_mode_log = enable_mode_log
        self._did_log_forward = False

        self.embedding = nn.Embedding(self.max_aaa_types + 1, feat_dim)
        self.distance = Distance(cutoff, max_num_neighbors=self.max_num_neighbors)

        # ---------------- vismp ----------------
        if self.atom_encoder_type == "vismp":
            self.sphere = Sphere(l=lmax)
            self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
            self.neighbor_embedding = NeighborEmbedding(feat_dim, num_rbf, cutoff, self.max_aaa_types)
            self.edge_embedding = EdgeEmbedding(num_rbf, feat_dim)

            self.vis_mp_layers = nn.ModuleList()
            vis_mp_kwargs = dict(
                num_heads=8,
                hidden_channels=feat_dim,
                cutoff=cutoff,
                vecnorm_type="none",
                trainable_vecnorm=False,
            )
            for i in range(num_layers):
                if i < num_layers - 1:
                    self.vis_mp_layers.append(ViS_MP(last_layer=False, **vis_mp_kwargs))
                else:
                    self.vis_mp_layers.append(ViS_MP(last_layer=True, **vis_mp_kwargs))

            self.out_norm = nn.LayerNorm(feat_dim)
            self.vec_out_norm = VecLayerNorm(feat_dim, trainable=False, norm_type="none")

        # ---------------- schnet ----------------
        elif self.atom_encoder_type == "schnet":
            self.schnet_rbf = GaussianRBFExpansion(
                cutoff=cutoff,
                num_rbf=num_rbf,
                trainable=trainable_rbf,
                start=0.0,
            )
            self.schnet_layers = nn.ModuleList([
                SchNetInteractionBlock(
                    hidden_channels=feat_dim,
                    num_rbf=num_rbf,
                    num_filters=schnet_num_filters if schnet_num_filters is not None else feat_dim,
                    activation=schnet_activation,
                )
                for _ in range(num_layers)
            ])
            self.out_norm = nn.LayerNorm(feat_dim)

        # ---------------- painn ----------------
        elif self.atom_encoder_type == "painn":
            self.painn_edge_size = painn_edge_size
            self.painn_pad_to_8 = painn_pad_to_8

            self.painn_message_layers = nn.ModuleList([
                PainnMessageBlock(
                    node_size=feat_dim,
                    edge_size=painn_edge_size,
                    cutoff=cutoff,
                )
                for _ in range(num_layers)
            ])
            self.painn_update_layers = nn.ModuleList([
                PainnUpdateBlock(node_size=feat_dim)
                for _ in range(num_layers)
            ])

            self.out_norm = nn.LayerNorm(feat_dim)
            self.vec_out_norm = VecLayerNorm(feat_dim, trainable=False, norm_type="none")

        else:
            raise ValueError(
                f"Unknown atom_encoder_type={atom_encoder_type}. "
                f"Supported: ['vismp', 'schnet', 'painn']"
            )

        self.reset_parameters()
        self._log_mode_once(
            num_layers=num_layers,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            schnet_num_filters=schnet_num_filters,
            painn_edge_size=painn_edge_size,
            painn_pad_to_8=painn_pad_to_8,
        )

    # ------------------------------------------------------------
    # logging
    # ------------------------------------------------------------
    def _log_mode_once(
        self,
        num_layers,
        num_rbf,
        trainable_rbf,
        schnet_num_filters,
        painn_edge_size,
        painn_pad_to_8,
    ):
        if not self.enable_mode_log:
            return

        if self.atom_encoder_type == "vismp":
            logger.info(
                "[AtomEmbedding] mode=vismp | cutoff=%.3f | num_layers=%d | num_rbf=%d | trainable_rbf=%s | "
                "output_vec_channels=8 | sphere=yes | neighbor_embedding=yes | edge_embedding=yes | edge_update=yes",
                self.cutoff, num_layers, num_rbf, str(trainable_rbf)
            )
        elif self.atom_encoder_type == "schnet":
            logger.info(
                "[AtomEmbedding] mode=schnet | cutoff=%.3f | num_layers=%d | num_rbf=%d | trainable_rbf=%s | "
                "num_filters=%s | output_vec_channels=8(padded_zero_vec) | sphere=no | neighbor_embedding=no | edge_embedding=no | edge_update=no",
                self.cutoff, num_layers, num_rbf, str(trainable_rbf), str(schnet_num_filters)
            )
        elif self.atom_encoder_type == "painn":
            logger.info(
                "[AtomEmbedding] mode=painn | cutoff=%.3f | num_layers=%d | edge_size=%d | "
                "output_vec_channels=%s | sphere=no | neighbor_embedding=no | edge_embedding=no | edge_update=no",
                self.cutoff, num_layers, painn_edge_size,
                "8(padded_from_3)" if painn_pad_to_8 else "3"
            )

    def _log_forward_once(self, aaa_feat, vec, mask_atoms, edge_index, raw_vec_shape=None):
        if (not self.enable_mode_log) or self._did_log_forward:
            return
        self._did_log_forward = True

        raw_vec_msg = ""
        if raw_vec_shape is not None:
            raw_vec_msg = f" | node_vector_raw={tuple(raw_vec_shape)}"

        logger.info(
            "[AtomEmbedding:forward] mode=%s | aaa_feat=%s%s | vec_out=%s | mask_atoms=%s | num_edges=%d",
            self.atom_encoder_type,
            tuple(aaa_feat.shape),
            raw_vec_msg,
            tuple(vec.shape),
            tuple(mask_atoms.shape),
            int(edge_index.size(1)),
        )

    # ------------------------------------------------------------
    # common utils
    # ------------------------------------------------------------
    def reset_parameters(self):
        self.embedding.reset_parameters()

        if self.atom_encoder_type == "vismp":
            self.distance_expansion.reset_parameters()
            self.neighbor_embedding.reset_parameters()
            self.edge_embedding.reset_parameters()
            for layer in self.vis_mp_layers:
                layer.reset_parameters()
            self.out_norm.reset_parameters()
            self.vec_out_norm.reset_parameters()

        elif self.atom_encoder_type == "schnet":
            self.schnet_rbf.reset_parameters()
            for layer in self.schnet_layers:
                layer.reset_parameters()
            self.out_norm.reset_parameters()

        elif self.atom_encoder_type == "painn":
            for layer in self.painn_message_layers:
                layer.reset_parameters()
            for layer in self.painn_update_layers:
                layer.reset_parameters()
            self.out_norm.reset_parameters()
            self.vec_out_norm.reset_parameters()

    def _prepare_atom_scalar_input(self, pos_atoms, mask_atoms, sequence_mask=None):
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        heavy_atoms = torch.arange(mask_atoms.size(-1), device=mask_atoms.device)
        aaa = torch.where(
            mask_atoms,
            heavy_atoms[None, None, :],
            torch.full(
                mask_atoms.shape,
                fill_value=self.max_aaa_types - 1,
                dtype=torch.long,
                device=mask_atoms.device,
            ),
        )

        if sequence_mask is not None:
            aaa = torch.where(
                (sequence_mask[:, :, None]) | (aaa == self.max_aaa_types - 1),
                aaa,
                torch.full(
                    mask_atoms.shape,
                    fill_value=self.max_aaa_types,
                    dtype=torch.long,
                    device=mask_atoms.device,
                ),
            )

        N, L, A = aaa.size()
        aaa_flat = aaa.view(N * L * A)
        aaa_flat = torch.clamp(aaa_flat, max=self.max_aaa_types)

        aaa_feat = self.embedding(aaa_flat)
        aaa_feat = aaa_feat.view(N, L, A, -1)
        aaa_feat = torch.where(
            (aaa.view(N, L, A) != (self.max_aaa_types - 1)).unsqueeze(-1),
            aaa_feat,
            torch.zeros_like(aaa_feat, device=mask_atoms.device),
        )
        aaa_feat = aaa_feat.view(N * L * A, -1)

        return pos_atoms, mask_atoms, aaa_flat, aaa_feat, N, L, A

    def _build_edges(self, pos_atoms, mask_atoms, structure_mask=None):
        edge_index, edge_weight, edge_vec = self.distance(pos_atoms, mask_atoms)

        if structure_mask is not None:
            N, L, A = mask_atoms.size()

            batch_indices = torch.clamp(
                torch.div(edge_index[0], (L * A), rounding_mode="trunc"),
                0,
                N - 1,
            )
            residue_indices_1 = torch.clamp(
                torch.div((edge_index[0] % (L * A)), A, rounding_mode="trunc"),
                0,
                L - 1,
            )
            residue_indices_2 = torch.clamp(
                torch.div((edge_index[1] % (L * A)), A, rounding_mode="trunc"),
                0,
                L - 1,
            )

            valid_edges = (
                structure_mask[batch_indices, residue_indices_1]
                & structure_mask[batch_indices, residue_indices_2]
            )

            edge_index = edge_index[:, valid_edges]
            edge_weight = edge_weight[valid_edges]
            edge_vec = edge_vec[valid_edges]

        return edge_index, edge_weight, edge_vec

    @staticmethod
    def _pad_vec3_to_vec8(vec3: torch.Tensor):
        """
        vec3: (num_atoms, 3, hidden)
        return vec8: (num_atoms, 8, hidden)
        """
        num_atoms, _, hidden = vec3.shape
        vec8 = torch.zeros(num_atoms, 8, hidden, device=vec3.device, dtype=vec3.dtype)
        vec8[:, :3, :] = vec3
        return vec8

    # ------------------------------------------------------------
    # branch: vismp
    # ------------------------------------------------------------
    def _forward_vismp(self, aaa_flat, aaa_feat, edge_index, edge_weight, edge_vec, mask_atoms):
        edge_attr = self.distance_expansion(edge_weight)

        norm = torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        edge_vec = edge_vec / norm

        edge_vec = self.sphere(edge_vec)
        aaa_feat = self.neighbor_embedding(aaa_flat, aaa_feat, edge_index, edge_weight, edge_attr)
        edge_attr = self.edge_embedding(edge_index, edge_attr, aaa_feat)

        vec = torch.zeros(
            aaa_feat.size(0),
            ((self.lmax + 1) ** 2) - 1,
            aaa_feat.size(1),
            device=mask_atoms.device,
        )

        for attn in self.vis_mp_layers:
            dx, dvec, dedge_attr = attn(
                aaa_feat, vec, edge_index, edge_weight, edge_attr, edge_vec
            )
            aaa_feat = aaa_feat + dx
            vec = vec + dvec
            if dedge_attr is not None:
                edge_attr = edge_attr + dedge_attr

        aaa_feat = self.out_norm(aaa_feat)
        vec = self.vec_out_norm(vec)

        self._log_forward_once(
            aaa_feat=aaa_feat,
            vec=vec,
            mask_atoms=mask_atoms,
            edge_index=edge_index,
            raw_vec_shape=vec.shape,
        )

        return {
            "aaa_feat": aaa_feat,
            "vec": vec,
            "mask_atoms": mask_atoms,
        }

    # ------------------------------------------------------------
    # branch: schnet
    # ------------------------------------------------------------
    def _forward_schnet(self, aaa_feat, edge_index, edge_weight, mask_atoms):
        edge_attr = self.schnet_rbf(edge_weight)

        x = aaa_feat
        for layer in self.schnet_layers:
            x = layer(x, edge_index, edge_attr)

        x = self.out_norm(x)

        vec = torch.zeros(
            x.size(0),
            ((self.lmax + 1) ** 2) - 1,
            x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        self._log_forward_once(
            aaa_feat=x,
            vec=vec,
            mask_atoms=mask_atoms,
            edge_index=edge_index,
            raw_vec_shape=(x.size(0), 0, x.size(1)),
        )

        return {
            "aaa_feat": x,
            "vec": vec,
            "mask_atoms": mask_atoms,
        }

    # ------------------------------------------------------------
    # branch: painn
    # ------------------------------------------------------------
    def _forward_painn(self, aaa_feat, edge_index, edge_weight, edge_vec, mask_atoms):
        """
        PaiNN branch:
        - keep PaiNN internal vector state as (num_atoms, 3, hidden)
        - zero-pad to (num_atoms, 8, hidden) only at output, so ga.py stays unchanged
        """
        x = aaa_feat

        # IMPORTANT:
        # Current Distance returns edge_vec = pos[edge_index[0]] - pos[edge_index[1]].
        # PaiNN message uses edge[:,0] as receiver i, edge[:,1] as sender j, and expects edge_diff = R_j - R_i.
        # So we use -edge_vec here.
        edge = torch.stack([edge_index[0], edge_index[1]], dim=1)   # (E, 2)
        edge_diff = -edge_vec                                        # (E, 3)
        edge_dist = edge_weight.clamp(min=1e-8)

        vec3 = torch.zeros(
            x.size(0), 3, x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        for msg_layer, upd_layer in zip(self.painn_message_layers, self.painn_update_layers):
            x, vec3 = msg_layer(x, vec3, edge, edge_diff, edge_dist)
            x, vec3 = upd_layer(x, vec3)

        x = self.out_norm(x)
        vec3 = self.vec_out_norm(vec3)

        if self.painn_pad_to_8:
            vec = self._pad_vec3_to_vec8(vec3)
        else:
            vec = vec3

        self._log_forward_once(
            aaa_feat=x,
            vec=vec,
            mask_atoms=mask_atoms,
            edge_index=edge_index,
            raw_vec_shape=vec3.shape,
        )

        return {
            "aaa_feat": x,
            "vec": vec,
            "mask_atoms": mask_atoms,
        }

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(
        self,
        aa,
        res_nb,
        chain_nb,
        pos_atoms,
        mask_atoms,
        fragment_type,
        structure_mask=None,
        sequence_mask=None,
    ):
        """
        Args:
            aa:         (N, L)
            res_nb:     (N, L)
            chain_nb:   (N, L)
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            fragment_type:  (N, L)
            structure_mask: (N, L)
            sequence_mask:  (N, L)
        """
        pos_atoms, mask_atoms, aaa_flat, aaa_feat, N, L, A = self._prepare_atom_scalar_input(
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
            sequence_mask=sequence_mask,
        )

        edge_index, edge_weight, edge_vec = self._build_edges(
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
            structure_mask=structure_mask,
        )

        if self.atom_encoder_type == "vismp":
            return self._forward_vismp(
                aaa_flat=aaa_flat,
                aaa_feat=aaa_feat,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_vec=edge_vec,
                mask_atoms=mask_atoms,
            )

        elif self.atom_encoder_type == "schnet":
            return self._forward_schnet(
                aaa_feat=aaa_feat,
                edge_index=edge_index,
                edge_weight=edge_weight,
                mask_atoms=mask_atoms,
            )

        elif self.atom_encoder_type == "painn":
            return self._forward_painn(
                aaa_feat=aaa_feat,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_vec=edge_vec,
                mask_atoms=mask_atoms,
            )

        else:
            raise RuntimeError(f"Unexpected atom_encoder_type={self.atom_encoder_type}")
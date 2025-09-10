'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-10-24 06:50:26
LastEditors: Patrick221215 1427584833@qq.com
LastEditTime: 2025-09-10 08:07:03
FilePath: /cjm/project/diffab/diffab/modules/encoders/atom.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
    NeighborEmbedding,
    EdgeEmbedding,
    VecLayerNorm,
    CosineCutoff,
    act_class_mapping
)

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
                    #1, 
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
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
    

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
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)
        
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
        
        # 使用较小的 gain 来缩小初始化权重
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
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
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
        #dx = torch.tanh(vec_dot) * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
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
    
    
class AtomEmbedding(nn.Module):
    def __init__(self, 
                 feat_dim,
                 max_num_atoms,
                 max_aaa_types=6, 
                 max_num_neighbors=16,
                 lmax=2,
                 cutoff=5.0,
                 num_layers=6,
                 num_rbf=32,
                 trainable_rbf=False,
                 ):
        super().__init__()
        
        self.lmax = lmax
        self.hidden_channels = feat_dim
        self.max_aaa_types = max_aaa_types
        self.cutoff = cutoff
        #self.max_num_atoms = max_num_atoms
        self.max_num_atoms = 5
        self.max_num_neighbors = max_num_neighbors
        
        self.embedding = nn.Embedding(self.max_aaa_types + 1, feat_dim)
        #nn.init.normal_(self.embedding.weight, mean=0, std=1)

        self.distance = Distance(cutoff, max_num_neighbors=self.max_num_neighbors)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(feat_dim, num_rbf, cutoff, self.max_aaa_types).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, feat_dim).jittable()
        
        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=8, 
            hidden_channels=feat_dim, 
            cutoff=cutoff, 
            vecnorm_type='none', 
            trainable_vecnorm=False
        )
        for i in range(num_layers):
            if i < num_layers - 1:
                self.vis_mp_layers.append(ViS_MP(last_layer=False, **vis_mp_kwargs).jittable())
            else:
                self.vis_mp_layers.append(ViS_MP(last_layer=True, **vis_mp_kwargs).jittable())

        self.out_norm = nn.LayerNorm(feat_dim)
        self.vec_out_norm = VecLayerNorm(feat_dim, trainable=False, norm_type='none')
        
        # self.output_model = EquivariantScalar(feat_dim)
        
        self.reset_parameters()
        
        
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()
        # self.output_model.reset_parameters()
        
    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms, fragment_type, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa:         (N, L).
            res_nb:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            fragment_type:  (N, L).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask:  (N, L), mask out unknown amino acids to generate.
        """
        
        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
            
        #Atoms identity features
        heavy_atoms = torch.arange(mask_atoms.size(-1), device=mask_atoms.device)
        aaa = torch.where(mask_atoms,heavy_atoms[None, None, :],torch.full(mask_atoms.shape, fill_value=self.max_aaa_types - 1, dtype=torch.long, device=mask_atoms.device))
        if sequence_mask is not None:
            #Avoid data leakage at training time 
            aaa = torch.where((sequence_mask[:, :, None]) | (aaa == self.max_aaa_types - 1),aaa,torch.full(mask_atoms.shape, fill_value=self.max_aaa_types, dtype=torch.long, device=mask_atoms.device))

        N, L, A = aaa.size()
        aaa = aaa.view(N * L * A)
        aaa = torch.clamp(aaa, max=self.max_aaa_types)

        aaa_feat = self.embedding(aaa) #(N, L, feat)
        
        # if torch.isnan(self.embedding.weight).any():
        #     import ipdb; ipdb.set_trace()
        #     it  = 0 
        #     print(f"NaN detected in embedding.weight at iteration {0}")
        #     torch.save({
        #     'embedding_weight': self.embedding.weight.clone().detach(),
        #     'pos_atoms': pos_atoms,
        #     'iteration': 0
        # }, f"nan_debug_iteration_{0}.pth")
        #     raise RuntimeError("NaN detected in embedding weight")
        # 氨基酸不存在的原子位置设为零
        # 将 aaa_feat 恢复形状
        #aaa_feat = torch.where(torch.isnan(aaa_feat), torch.zeros_like(aaa_feat), aaa_feat)
        aaa_feat = aaa_feat.view(N, L, A, -1)  # (N, L, A, feat_dim)
        aaa_feat = torch.where((aaa.view(N, L, A) != (self.max_aaa_types - 1)).unsqueeze(-1), aaa_feat, torch.zeros_like(aaa_feat, device=mask_atoms.device))
        aaa_feat = aaa_feat.view(N * L * A, -1)
        
        #distance embedding
        edge_index, edge_weight, edge_vec = self.distance(pos_atoms,mask_atoms)

        # edge_attr = self.distance_expansion(edge_weight)
        # # edge_vec = edge_vec/ torch.norm(edge_vec, dim=1).unsqueeze(1)
        # norm = torch.norm(edge_vec, dim=1).unsqueeze(1)
        # norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # 将零范数替换为1，以避免除以零
        # edge_vec = edge_vec / norm

        # edge_vec = self.sphere(edge_vec)
        # aaa_feat = self.neighbor_embedding(aaa, aaa_feat, edge_index, edge_weight, edge_attr)
        # edge_attr = self.edge_embedding(edge_index, edge_attr, aaa_feat)
        
        # vec = torch.zeros(aaa_feat.size(0), ((self.lmax + 1) ** 2) - 1, aaa_feat.size(1), device=mask_atoms.device)
        # #return aaa_feat, vec, edge_index, edge_weight, edge_attr, edge_vec
        # atom_feat = {
        #     'aaa_feat': aaa_feat,
        #     'vec': vec,
        #     'edge_index': edge_index,
        #     'edge_weight': edge_weight,
        #     'edge_attr': edge_attr,
        #     'edge_vec': edge_vec,
        #     'mask_atoms': mask_atoms,
        # }
        
        # # 遍历 atom_feat 中的每个键和值
        # for key, value in atom_feat.items():
        #     if torch.is_tensor(value):
        #         nan_mask = torch.isnan(value)
        #         if torch.any(nan_mask):
        #             nan_indices = torch.nonzero(nan_mask)
        #             print(f"Found NaN values in '{key}' at the following indices:")
        #             print(nan_indices)
        #         else:
        #             print(f"No NaN values found in '{key}'.")
                    
        # Apply structure_mask to edge_index and related edge attributes
        if structure_mask is not None:
            # 获取每个边的原子所属的残基索引和样本索引
            batch_indices = torch.clamp(torch.div(edge_index[0], (L * A), rounding_mode='trunc'), 0, N - 1)  # 限制在 0 到 N-1
            residue_indices_1 = torch.clamp(torch.div((edge_index[0] % (L * A)), A, rounding_mode='trunc'), 0, L - 1)  # 第一原子的残基索引
            residue_indices_2 = torch.clamp(torch.div((edge_index[1] % (L * A)), A, rounding_mode='trunc'), 0, L - 1)  # 第二原子的残基索引

            # 根据结构遮掩获取每条边的有效性
            valid_edges = structure_mask[batch_indices, residue_indices_1] & structure_mask[batch_indices, residue_indices_2]

            # 筛选有效的边
            edge_index = edge_index[:, valid_edges]
            edge_weight = edge_weight[valid_edges]
            edge_vec = edge_vec[valid_edges]
            #edge_attr = edge_attr[valid_edges]
            

        edge_attr = self.distance_expansion(edge_weight)
        # edge_vec = edge_vec/ torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # 将零范数替换为1，以避免除以零
        edge_vec = edge_vec / norm

        edge_vec = self.sphere(edge_vec)
        aaa_feat = self.neighbor_embedding(aaa, aaa_feat, edge_index, edge_weight, edge_attr)
        edge_attr = self.edge_embedding(edge_index, edge_attr, aaa_feat)
        
        vec = torch.zeros(aaa_feat.size(0), ((self.lmax + 1) ** 2) - 1, aaa_feat.size(1), device=mask_atoms.device)

        for attn in self.vis_mp_layers:
            
            dx, dvec, dedge_attr = attn(aaa_feat, vec, edge_index, edge_weight, edge_attr, edge_vec)
            # 更新原子特征、方向向量和边特征
            aaa_feat = aaa_feat + dx
            vec = vec + dvec
            if dedge_attr is not None:
                edge_attr = edge_attr + dedge_attr
            
        aaa_feat = self.out_norm(aaa_feat)
        vec = self.vec_out_norm(vec)
        
        # aaa_feat = self.output_model.pre_reduce(aaa_feat, vec)
        
        
        atom_feat = {
            'aaa_feat': aaa_feat,
            'vec': vec,
            'mask_atoms': mask_atoms,
        }
        
        # # 遍历 atom_feat 中的每个键和值
        # for key, value in atom_feat.items():
        #     if torch.is_tensor(value):
        #         nan_mask = torch.isnan(value)
        #         if torch.any(nan_mask):
        #             nan_indices = torch.nonzero(nan_mask)
        #             print(f"Found NaN values in '{key}' at the following indices:")
        #             print(nan_indices)
        #         else:
        #             print(f"No NaN values found in '{key}'.")
        
        #         #for i, block in enumerate(self.blocks):
                
        return atom_feat
'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-10-24 06:50:26
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-11-01 09:11:18
FilePath: /cjm/project/AbMEGD/AbMEGD/modules/encoders/atom.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn


from AbMEGD.modules.common.layers import (
    Distance,
    Sphere, 
    ExpNormalSmearing,
    NeighborEmbedding,
    EdgeEmbedding
)

class AtomEmbedding(nn.Module):
    def __init__(self, 
                 feat_dim,
                 max_num_atoms,
                 max_aaa_types=16, 
                 max_num_neighbors=16,
                 lmax=2,
                 cutoff=5.0,
                 num_rbf=32,
                 trainable_rbf=False,
                 ):
        super().__init__()
        
        self.lmax = lmax
        self.hidden_channels = feat_dim
        self.max_aaa_types = max_aaa_types
        self.cutoff = cutoff
        self.max_num_atoms = max_num_atoms
        self.max_num_neighbors = max_num_neighbors
        
        self.embedding = nn.Embedding(self.max_aaa_types + 1, feat_dim)
        self.distance = Distance(cutoff, max_num_neighbors=self.max_num_neighbors)
        self.sphere = Sphere(l=lmax)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable_rbf)
        self.neighbor_embedding = NeighborEmbedding(feat_dim, num_rbf, cutoff, self.max_aaa_types).jittable()
        self.edge_embedding = EdgeEmbedding(num_rbf, feat_dim).jittable()
        
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
        aaa_feat = self.embedding(aaa) #(N, L, feat)
        # 氨基酸不存在的原子位置设为零
        # 将 aaa_feat 恢复形状
        aaa_feat = aaa_feat.view(N, L, A, -1)  # (N, L, A, feat_dim)
        aaa_feat = torch.where((aaa.view(N, L, A) != (self.max_aaa_types - 1)).unsqueeze(-1), aaa_feat, torch.zeros_like(aaa_feat, device=mask_atoms.device))
        aaa_feat = aaa_feat.view(N * L * A, -1)
        
        #distance embedding
        edge_index, edge_weight, edge_vec = self.distance(pos_atoms,mask_atoms)

        edge_attr = self.distance_expansion(edge_weight)
        # edge_vec = edge_vec/ torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # 将零范数替换为1，以避免除以零
        edge_vec = edge_vec / norm

        edge_vec = self.sphere(edge_vec)
        aaa_feat = self.neighbor_embedding(aaa, aaa_feat, edge_index, edge_weight, edge_attr)
        edge_attr = self.edge_embedding(edge_index, edge_attr, aaa_feat)
        # Apply structure_mask to edge_index and related edge attributes
        if structure_mask is not None:
            # # 获取每个边的原子所属的残基索引和样本索引
            # batch_indices = torch.clamp(edge_index[0] // (L * A), 0, N - 1)  # 限制在 0 到 N-1
            # residue_indices_1 = torch.clamp((edge_index[0] % (L * A)) // A, 0, L - 1)  # 第一原子的残基索引
            # residue_indices_2 = torch.clamp((edge_index[1] % (L * A)) // A, 0, L - 1)  # 第二原子的残基索引

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
            edge_attr = edge_attr[valid_edges]
            
        # if structure_mask is not None:
        #     # Avoid data leakage at training time
        #     pos_atoms = pos_atoms * structure_mask[:, :, None,None]
        

        
        vec = torch.zeros(aaa_feat.size(0), ((self.lmax + 1) ** 2) - 1, aaa_feat.size(1), device=mask_atoms.device)
        #return aaa_feat, vec, edge_index, edge_weight, edge_attr, edge_vec
        atom_feat = {
            'aaa_feat': aaa_feat,
            'vec': vec,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'edge_attr': edge_attr,
            'edge_vec': edge_vec,
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
        
        return atom_feat
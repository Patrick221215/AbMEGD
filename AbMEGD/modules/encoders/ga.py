import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from AbMEGD.modules.common.layers import mask_zero, LayerNorm
from AbMEGD.modules.common.geometry import global_to_local, local_to_global, normalize_vector

def _alpha_from_logits(logits, mask, inf=1e5):
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)
    logits = torch.where(mask_pair, logits, logits - inf)
    alpha = torch.softmax(logits, dim=2)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha

def _heads(x, n_heads, n_ch):
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, atom_feat_dim, vec_dim, vec_feat_dim, 
                 value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
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

        self.vec_to_scalar = nn.Linear(self.vec_dim, self.vec_feat_dim, bias=True)
        self.proj_atom_query = nn.Linear(self.atom_feat_dim + self.vec_feat_dim, self.num_heads * self.query_key_dim, bias=True)
        self.proj_atom_key   = nn.Linear(self.atom_feat_dim + self.vec_feat_dim, self.num_heads * self.query_key_dim, bias=True)
        self.proj_atom_value = nn.Linear(self.atom_feat_dim + self.vec_feat_dim, self.num_heads * self.value_dim, bias=True)
        
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_feat_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_feat_dim, value_dim * num_heads, bias=bias)
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=bias)
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)
        self.proj_query_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_feat_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_feat_dim, num_value_points * num_heads * 3, bias=bias)
        
        in_features = (num_heads * pair_feat_dim) + (num_heads * value_dim) + \
                      (num_heads * num_value_points * 7) + (num_heads * value_dim)
        self.out_transform = nn.Linear(in_features=in_features, out_features=node_feat_dim)

        self.layer_norm_1 = LayerNorm(node_feat_dim)
        self.layer_norm_2 = LayerNorm(node_feat_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                                            nn.Linear(node_feat_dim, node_feat_dim))
    
    def _atom_logits(self, aaa_feat, vec, mask_atoms):
        # Input shapes from your debugger:
        # aaa_feat:   torch.Size([15840, 64])
        # vec:        torch.Size([15840, 8, 64])
        # mask_atoms: torch.Size([16, 198, 5])

        N, L, A = mask_atoms.shape
        atom_feat_dim = aaa_feat.shape[-1]
        
        #import ipdb; ipdb.set_trace()
        vec_dim_expected = self.vec_to_scalar.in_features # This should be 512
        vec_flattened = vec.reshape(-1, vec_dim_expected) # Shape: (15840, 512)

        # Reshape flattened features back to their structured (N, L, A, Dim) form
        aaa_feat_reshaped = aaa_feat.view(N, L, A, atom_feat_dim)
        vec_reshaped = vec_flattened.view(N, L, A, vec_dim_expected)

        # Now all tensors have consistent structured shapes for broadcasting
        mask_atoms_expanded = mask_atoms.unsqueeze(-1)
        valid_atom_counts = mask_atoms.sum(dim=2, keepdim=True).clamp(min=1e-8)
        
        # Aggregate atomic scalar features
        residue_atom_feat = (aaa_feat_reshaped * mask_atoms_expanded).sum(dim=2) / valid_atom_counts
        
        # Aggregate atomic vector features
        residue_vec_feat = (vec_reshaped * mask_atoms_expanded).sum(dim=2) / valid_atom_counts

        # Continue with the rest of your correct logic
        residue_vec_feat_scalar = self.vec_to_scalar(residue_vec_feat)
        atom_feat_combined = torch.cat([residue_atom_feat, residue_vec_feat_scalar], dim=-1)
        
        query_a = _heads(self.proj_atom_query(atom_feat_combined), self.num_heads, self.query_key_dim)
        key_a = _heads(self.proj_atom_key(atom_feat_combined), self.num_heads, self.query_key_dim)
        
        logits_atom = (query_a.unsqueeze(2) * key_a.unsqueeze(1) * (1 / np.sqrt(self.query_key_dim))).sum(-1)
        return logits_atom, atom_feat_combined

    
    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)
        logits_node = (query_l.unsqueeze(2) * key_l.unsqueeze(1) * (1 / np.sqrt(self.query_key_dim))).sum(-1)
        return logits_node

    def _pair_logits(self, z):
        return self.proj_pair_bias(z)

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points, 3)
        query_points = local_to_global(R, t, query_points)
        query_s = query_points.reshape(N, L, self.num_heads, -1)
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points, 3)
        key_points = local_to_global(R, t, key_points)
        key_s = key_points.reshape(N, L, self.num_heads, -1)
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2)
        return logits_spatial

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
        
    def _atom_aggregation(self, alpha, atom_feat_combined):
        value_a = _heads(self.proj_atom_value(atom_feat_combined), self.num_heads, self.value_dim)
        feat_atom = alpha.unsqueeze(-1) * value_a.unsqueeze(1)
        feat_atom = feat_atom.sum(dim=2)
        return feat_atom.reshape(atom_feat_combined.shape[0], atom_feat_combined.shape[1], -1)

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points, 3))
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * value_points.unsqueeze(1)
        aggr_points = aggr_points.sum(dim=2)
        feat_points = global_to_local(R, t, aggr_points)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = F.normalize(feat_points, dim=-1, eps=1e-4)
        feat_spatial = torch.cat([feat_points.reshape(N, L, -1), feat_distance.reshape(N, L, -1), feat_direction.reshape(N, L, -1)], dim=-1)
        return feat_spatial
    
    def forward(self, R, t, x, z, aaa_feat, vec, mask, mask_atoms):
        logits_atom, atom_feat_combined = self._atom_logits(aaa_feat, vec, mask_atoms)
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        
        logits_sum = logits_node + logits_pair + logits_spatial + logits_atom
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 4), mask)

        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_atom = self._atom_aggregation(alpha, atom_feat_combined)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial, feat_atom], dim=-1))
        
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm_1(x + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated

class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, ga_block_opt={}):
        super(GAEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            GABlock(
                node_feat_dim=node_feat_dim, 
                pair_feat_dim=pair_feat_dim, 
                atom_feat_dim=int(node_feat_dim*0.5),
                vec_dim=512,
                vec_feat_dim=int(node_feat_dim*0.5),
                **ga_block_opt
            ) for _ in range(num_layers)
        ])


    def forward(self, R, t, atom_feat, res_feat, pair_feat, mask):
        aaa_feat = atom_feat['aaa_feat']
        vec = atom_feat['vec']
        mask_atoms = atom_feat['mask_atoms']
        
        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, aaa_feat, vec, mask, mask_atoms)
            
        return res_feat
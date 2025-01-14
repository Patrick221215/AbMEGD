import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing




class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes) # (N, L)
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[...,None]  # (N, L, C)
    return y


class DistanceToBins(nn.Module):

    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot=False):
        super().__init__()
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.num_bins = num_bins
        self.use_onehot = use_onehot

        if use_onehot:
            offset = torch.linspace(dist_min, dist_max, self.num_bins)
        else:
            offset = torch.linspace(dist_min, dist_max, self.num_bins-1)    # 1 overflow flag
            self.coeff = -0.5 / ((offset[1] - offset[0]) * 0.2).item() ** 2  # `*0.2`: makes it not too blurred
        self.register_buffer('offset', offset)

    @property
    def out_channels(self):
        return self.num_bins 

    def forward(self, dist, dim, normalize=True):
        """
        Args:
            dist:   (N, *, 1, *)
        Returns:
            (N, *, num_bins, *)
        """
        assert dist.size()[dim] == 1
        offset_shape = [1] * len(dist.size())
        offset_shape[dim] = -1

        if self.use_onehot:
            diff = torch.abs(dist - self.offset.view(*offset_shape))  # (N, *, num_bins, *)
            bin_idx = torch.argmin(diff, dim=dim, keepdim=True)  # (N, *, 1, *)
            y = torch.zeros_like(diff).scatter_(dim=dim, index=bin_idx, value=1.0)
        else:
            overflow_symb = (dist >= self.dist_max).float()  # (N, *, 1, *)
            y = dist - self.offset.view(*offset_shape)  # (N, *, num_bins-1, *)
            y = torch.exp(self.coeff * torch.pow(y, 2))  # (N, *, num_bins-1, *)
            y = torch.cat([y, overflow_symb], dim=dim)  # (N, *, num_bins, *)
            if normalize:
                y = y / y.sum(dim=dim, keepdim=True)

        return y


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))
    
    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class AngularEncoding(nn.Module):

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2*2*self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )


  
class Distance(nn.Module): 
    def __init__(self, cutoff, max_num_neighbors=16):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos_atoms, mask_atoms):
        """
        Args:
            pos_atoms: Tensor of shape (N, L, A, 3), where
                - N is the number of samples,
                - L is the number of residues per sample,
                - A is the number of atoms per residue,
                - 3 represents the 3D coordinates of each atom.
        
        Returns:
            edge_index: Edge indices of the neighbors.
            edge_weight: Distances between neighbors.
            edge_vec: Vector difference between neighboring atoms' positions.
        """
        
        pos_atoms = torch.where(mask_atoms[..., None], pos_atoms, torch.full_like(pos_atoms, float('nan'),device = pos_atoms.device))
        N, L, A, _ = pos_atoms.size()
        # Flatten pos_atoms to a 2D tensor of shape (N*L*A, 3)
        pos = pos_atoms.view(N * L * A, 3)
        
        # Create batch indices indicating which sample each atom belongs to
        # This is important for processing multiple samples correctly
        batch = torch.arange(N).repeat_interleave(L * A).to(pos_atoms.device)
        # residue_index 表示每个原子所属的残基索引 (L)
        residue_index = torch.arange(L).repeat_interleave(A).repeat(N).to(pos_atoms.device)  # shape: (N * L * A)
        # atom_index 表示每个原子在残基中的位置 (A)
        atom_index = torch.arange(A).repeat(L * N).to(pos_atoms.device)  # shape: (N * L * A)

        # Calculate edge_index using the flattened pos and batch information
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False, max_num_neighbors=self.max_num_neighbors)
        
        valid_edges = ~torch.isnan(pos[edge_index[0]]).any(dim=-1) & ~torch.isnan(pos[edge_index[1]]).any(dim=-1)
        edge_index = edge_index[:, valid_edges]

        # Calculate edge_vec using the flattened pos
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        
        # Calculate edge_weight as the norm of edge_vec
        edge_weight = torch.norm(edge_vec, dim=-1)
        # # Filter edges to ensure they are within the same residue
        # same_residue_mask = residue_index[edge_index[0]] == residue_index[edge_index[1]]
        # edge_index = edge_index[:, same_residue_mask]

        # # Calculate edge_vec and edge_weight based on the filtered edge_index
        # edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        # edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec

class Sphere(nn.Module):
    
    def __init__(self, l=2):
        super(Sphere, self).__init__()
        self.l = l
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)

class CosineCutoff(nn.Module):
    
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs

class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z + 1, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        
        
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        assert z.max() < self.embedding.num_embeddings, f"z contains out-of-bound values: {z.max()} vs {self.embedding.num_embeddings}"

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        assert edge_index.max() < x_neighbors.size(0), f"edge_index contains out-of-bound values: {edge_index.max()} vs {x_neighbors.size(0)}"

        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W

    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)
    
    def aggregate(self, features, index):
        # no aggregate
        return features

class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * self.weight.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")

act_class_mapping = {"ssp": ShiftedSoftplus, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "swish": Swish}

import torch
from torch import nn
from torch.nn import Module

from einops import einsum

from torch_einops_utils import pack_with_inverse

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class PseudoProjector(Module):
    def __init__(
        self,
        dim,
        dim_lowrank,
        eps = 1e-10
    ):
        super().__init__()
        assert dim_lowrank < dim, 'low rank dimension must be smaller than model dimension'

        self.restrict = nn.Linear(dim, dim_lowrank, bias = False)
        self.prolong = nn.Linear(dim_lowrank, dim, bias = False)

        self.register_buffer('eye_with_eps', torch.eye(dim_lowrank) + eps)

    def forward(
        self,
        features
    ):
        features, inverse_pack = pack_with_inverse(features, '* d') # allow for any number of preceding dimension

        # P = Q (Q* Q)^−1 Q∗

        # follow notation in paper

        Q, Q_star = self.prolong.weight, self.restrict.weight

        coarse_grid_op = einsum(Q_star, Q, 'dc d, d ec -> ec dc') + self.eye_with_eps

        coarsened = self.restrict(features)

        # inverse

        u = torch.linalg.solve(coarse_grid_op, coarsened, left = False)

        projected = self.prolong(u)

        return inverse_pack(projected)

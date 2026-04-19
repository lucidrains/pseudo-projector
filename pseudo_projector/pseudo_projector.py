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

# classes

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

        self.register_buffer('eye_with_eps', torch.eye(dim_lowrank) + eps, persistent = False)

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

# with the original features as residual, with static or dynamic gate (alpha in paper)

class PseudoProjectorWithResidual(Module):
    def __init__(
        self,
        dim,
        dim_lowrank,
        learned_alpha = False,          # learned frac
        static_alpha = 0.5,             # fraction of output that is pseudo projected
        norm_before_proj_gate = False,  # norm before projecting to learned alpha
        per_feature = False,
        eps = 1e-10,
    ):
        super().__init__()

        self.pseudo_proj = PseudoProjector(dim, dim_lowrank, eps)

        self.learned_alpha = learned_alpha

        assert 0. <= static_alpha <= 1.
        self.static_alpha = static_alpha

        if learned_alpha:
            self.to_learned_alpha = nn.Sequential(
                nn.RMSNorm(dim) if norm_before_proj_gate else nn.Identity(),
                nn.Linear(dim, dim if per_feature else 1, bias = False),
                nn.Sigmoid()
            )

    @staticmethod
    def set_static_alpha_(
        network: Module,
        static_alpha: float
    ):
        # for manual scheduling during training

        assert 0. <= static_alpha <= 1.

        for module in network.modules():
            if (
                not isinstance(module, PseudoProjectorWithResidual) or
                module.learned_alpha
            ):
                continue

            module.static_alpha = static_alpha

    def forward(
        self,
        feats
    ):
        residual = feats

        projected = self.pseudo_proj(feats)

        # static or dynamic

        if self.learned_alpha:
            alpha = self.to_learned_alpha(residual)
        else:
            alpha = self.static_alpha

        # blended output

        return residual.lerp(projected, alpha) # alpha is fraction of the projected

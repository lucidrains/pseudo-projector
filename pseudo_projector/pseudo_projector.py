import torch
from torch import nn, linalg
from torch.nn import Module

from einops import einsum

from torch_einops_utils import pack_with_inverse

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def newton_schulz_inverse(
    mat,
    iters = 20,
    eps = 1e-10
):
    dim = mat.shape[-1]
    identity = torch.eye(dim, device = mat.device, dtype = mat.dtype)

    # ensure initial approximation is within the radius of convergence

    norm_1 = linalg.matrix_norm(mat, ord = 1, dim = (-2, -1), keepdim = True)
    norm_inf = linalg.matrix_norm(mat, ord = float('inf'), dim = (-2, -1), keepdim = True)

    scale = 1. / (norm_1 * norm_inf).clamp(min = eps)
    inverse_approx = scale * mat.transpose(-1, -2)

    # newton schulz iterations

    for _ in range(iters):
        inverse_approx = inverse_approx @ (2 * identity - mat @ inverse_approx)

    return inverse_approx

# classes

class PseudoProjector(Module):
    def __init__(
        self,
        dim,
        dim_lowrank,
        eps = 1e-10,
        use_newton_schulz = False,
        newton_schulz_iters = 10
    ):
        super().__init__()
        assert dim_lowrank < dim, 'low rank dimension must be smaller than model dimension'

        self.restrict = nn.Linear(dim, dim_lowrank, bias = False)
        self.prolong = nn.Linear(dim_lowrank, dim, bias = False)

        self.use_newton_schulz = use_newton_schulz
        self.newton_schulz_iters = newton_schulz_iters

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

        if self.use_newton_schulz:
            coarse_grid_op_inv = newton_schulz_inverse(coarse_grid_op, iters = self.newton_schulz_iters)
            u = coarsened @ coarse_grid_op_inv
        else:
            u = linalg.solve(coarse_grid_op, coarsened, left = False)

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
        use_newton_schulz = False,
        newton_schulz_iters = 100
    ):
        super().__init__()

        self.pseudo_proj = PseudoProjector(
            dim,
            dim_lowrank,
            eps = eps,
            use_newton_schulz = use_newton_schulz,
            newton_schulz_iters = newton_schulz_iters
        )

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

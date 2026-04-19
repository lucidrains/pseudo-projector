import pytest
param = pytest.mark.parametrize

import torch

def test_pseudo_projector():
    from pseudo_projector.pseudo_projector import PseudoProjector

    proj = PseudoProjector(512, 32)

    features = torch.randn(1, 1024, 512)

    out = proj(features)

    assert out.shape == features.shape

@param('learned_alpha', (False, True))
def test_pseudo_projector_with_residual(
    learned_alpha
):
    from pseudo_projector.pseudo_projector import PseudoProjectorWithResidual

    proj_with_residual = PseudoProjectorWithResidual(512, 32, learned_alpha)

    features = torch.randn(1, 1024, 512)

    out = proj_with_residual(features)

    assert out.shape == features.shape

    if not learned_alpha:
        PseudoProjectorWithResidual.set_static_alpha_(proj_with_residual, 0.1)
        assert proj_with_residual.static_alpha == 0.1

def test_newton_schulz_equivalent():
    from pseudo_projector.pseudo_projector import PseudoProjector

    proj_exact = PseudoProjector(512, 32, use_newton_schulz = False)
    proj_ns = PseudoProjector(512, 32, use_newton_schulz = True, newton_schulz_iters = 10)

    proj_ns.load_state_dict(proj_exact.state_dict())

    features = torch.randn(2, 1024, 512)

    out_exact = proj_exact(features)
    out_ns = proj_ns(features)

    assert torch.allclose(out_exact, out_ns, atol = 1e-4)

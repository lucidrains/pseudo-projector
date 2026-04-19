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

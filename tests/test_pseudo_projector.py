import torch

def test_pseudo_projector():
    from pseudo_projector.pseudo_projector import PseudoProjector

    proj = PseudoProjector(512, 32)

    features = torch.randn(1, 1024, 512)

    out = proj(features)

    assert out.shape == features.shape

## Pseudo-projector

Implementation of the [pseudo projector](https://arxiv.org/abs/2603.09815) proposed by Vitaly Bulgakov during his work applying transformers to medical records at Mass General Brigham Hospital

## Install

```shell
$ pip install pseudo-projector
```

## Usage

```python
import torch
from pseudo_projector import PseudoProjector

proj = PseudoProjector(dim = 64, dim_lowrank = 16)

feats = torch.randn(1, 8, 1024, 64) # any number of preceding dimensions

out = proj(feats)

assert feats.shape == out.shape
```

With the learned blending of original features with the coarsened ones

```python
import torch
from pseudo_projector import PseudoProjector

proj = PseudoProjectorWithResidual(dim = 64, dim_lowrank = 16, learned_alpha = True)

feats = torch.randn(1, 8, 1024, 64) # any number of preceding dimensions

out = proj(feats)

assert feats.shape == out.shape
```

## Citations

```bibtex
@misc{bulgakov2026correctiontransformerbasedmodelssmoothing,
    title   = {Correction of Transformer-Based Models with Smoothing Pseudo-Projector},
    author  = {Vitaly Bulgakov},
    year    = {2026},
    eprint  = {2603.09815},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.09815},
}
```

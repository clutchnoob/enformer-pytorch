<img src="./enformer.png" width="450px"></img>

## Enformer - Pytorch

Implementation of <a href="https://deepmind.com/blog/article/enformer">Enformer</a>, Deepmind's attention network for predicting gene expression, in Pytorch. The original tensorflow sonnet code can be found <a href="https://github.com/deepmind/deepmind-research/tree/master/enformer">here</a>.

## Install

```bash
$ pip install enformer-pytorch
```

## Usage

```python
import torch
from enformer_pytorch import Enformer, SEQUENCE_LENGTH

model = Enformer(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse= 1643),
    target_length = 896,
)

seq = torch.randint(0, 5, (1, SEQUENCE_LENGTH)) # for NAGCT
output = model(seq)

output['human'] # (1, 896, 5313)
output['mouse'] # (1, 896, 1643)
```

## Citations

```bibtex
@article {Avsec2021.04.07.438649,
    author  = {Avsec, {\v Z}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
    title   = {Effective gene expression prediction from sequence by integrating long-range interactions},
    elocation-id = {2021.04.07.438649},
    year    = {2021},
    doi     = {10.1101/2021.04.07.438649},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649},
    eprint  = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649.full.pdf},
    journal = {bioRxiv}
}
```

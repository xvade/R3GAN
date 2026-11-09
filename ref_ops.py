"""Force pure-PyTorch ops so inference works without nvcc/ninja (no custom CUDA plugin builds).
Import this before loading a network pickle."""
import functools
import torch.nn.functional as F
from R3GAN import FusedOperators as FO
from torch_utils.ops import upfirdn2d as U

def _bias_act_ref(self, x):
    b = self.Bias.to(x.dtype)
    y = x + (b.view(1, -1, 1, 1) if x.ndim > 2 else b.view(1, -1))
    return F.leaky_relu(y, 0.2)
FO.BiasedActivationCUDA.forward = _bias_act_ref

_orig = U.upfirdn2d
@functools.wraps(_orig)
def _upfirdn2d_ref(*a, **k):
    k['impl'] = 'ref'
    return _orig(*a, **k)
U.upfirdn2d = _upfirdn2d_ref

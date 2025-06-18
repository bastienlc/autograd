import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)
torch.manual_seed(42)

n = 5
m = 10


def test_reduce_sum_forward():
    shape = (n, m)
    a = np.random.randn(*shape)
    result = (Tensor.from_numpy(a).reduce_sum()).to_numpy()

    assert np.allclose(a.sum(), result, atol=1e-6, rtol=1e-6)


def test_reduce_sum_backward():
    shape = (n, m)

    # torch implementation
    a1 = torch.randn(*shape, requires_grad=True)
    b1 = a1.sum()
    b1.backward()

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = a2.reduce_sum()
    b2.backward(None)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())

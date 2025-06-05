import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)

n = 5
m = 10


def test_relu_forward():
    shape = (n, m)
    a = np.random.randn(*shape)
    result = (Tensor.from_numpy(a).relu()).to_numpy()

    assert np.allclose(a * (a > 0), result)


def test_relu_backward():
    shape = (n, m)

    # torch implementation
    a1 = torch.randn(*shape, requires_grad=True)
    b1 = a1.relu()
    grad1 = torch.ones_like(b1)
    b1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = a2.relu()
    grad2 = Tensor.from_torch(grad1)
    b2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())

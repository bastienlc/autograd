import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)
torch.manual_seed(42)

batch = 10
n = 5
m = 10


def test_broadcast_forward_1():
    shape = (n, m)
    a = np.random.randn(1)
    result = (Tensor.from_numpy(a).broadcast(shape)).to_numpy()

    assert np.allclose(np.broadcast_to(a, shape), result, atol=1e-6, rtol=1e-6)


def test_broadcast_forward_2():
    shape = (batch, n, m)
    a = np.random.randn(n, m)
    result = (Tensor.from_numpy(a).broadcast(shape)).to_numpy()

    assert np.allclose(np.broadcast_to(a, shape), result, atol=1e-6, rtol=1e-6)


def test_broadcast_backward_1():
    shape = (n, m)

    # torch implementation
    a1 = torch.randn(1, requires_grad=True)
    b1 = a1.broadcast_to(shape)
    grad1 = torch.ones_like(b1)
    b1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = a2.broadcast(shape)
    grad2 = Tensor.from_torch(grad1)
    b2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())


def test_broadcast_backward_2():
    shape = (batch, n, m)

    # torch implementation
    a1 = torch.randn(n, m, requires_grad=True)
    b1 = a1.broadcast_to(shape)
    grad1 = torch.ones_like(b1)
    b1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = a2.broadcast(shape)
    grad2 = Tensor.from_torch(grad1)
    b2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())

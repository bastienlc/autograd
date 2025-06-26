import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)
torch.manual_seed(42)

n = 5
m = 10


def test_mul_forward_1():
    shape = (n, m)
    a = np.random.randn(*shape)
    b = np.random.randn(*shape)
    result = (Tensor.from_numpy(a) * Tensor.from_numpy(b)).to_numpy()

    assert np.allclose(a * b, result, atol=1e-6, rtol=1e-6)


def test_mul_forward_2():
    a = np.random.randn(n, m)
    b = np.random.randn(m)
    result = (Tensor.from_numpy(a) * Tensor.from_numpy(b)).to_numpy()

    assert np.allclose(a * b[None, :], result, atol=1e-6, rtol=1e-6)


def test_mul_forward_3():
    shape = (n, m)
    a = np.random.randn(*shape)
    result = (Tensor.from_numpy(a) * Tensor.from_numpy(a)).to_numpy()

    assert np.allclose(a * a, result, atol=1e-6, rtol=1e-6)


def test_mul_backward_1():
    shape = (n, m)

    # torch implementation
    a1 = torch.randn(*shape, requires_grad=True)
    b1 = torch.randn(*shape, requires_grad=True)
    c1 = a1 * b1
    grad1 = torch.ones_like(c1)
    c1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = Tensor.from_torch(b1, requires_grad=True)

    c2 = a2 * b2
    grad2 = Tensor.from_torch(grad1)
    c2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())
    assert torch.allclose(b1.grad, b2.get_grad().to_torch())


def test_mul_backward_2():
    # torch implementation
    a1 = torch.randn((n, m), requires_grad=True)
    b1 = torch.randn((m,), requires_grad=True)
    c1 = a1 * b1
    grad1 = torch.ones_like(c1)
    c1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = Tensor.from_torch(b1, requires_grad=True)

    c2 = a2 * b2
    grad2 = Tensor.from_torch(grad1)
    c2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())
    assert torch.allclose(b1.grad, b2.get_grad().to_torch())


def test_mul_backward_3():
    shape = (n, m)

    # torch implementation
    a1 = torch.randn(*shape, requires_grad=True)
    b1 = a1 * a1
    grad1 = torch.ones_like(b1)
    b1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)

    b2 = a2 * a2
    grad2 = Tensor.from_torch(grad1)
    b2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())

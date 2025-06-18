import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)
torch.manual_seed(42)

batch = 10
n = 5
m = 10
p = 15


def test_matmul_forward_1():
    shape1 = (n, m)
    shape2 = (m, p)
    a = np.random.randn(*shape1)
    b = np.random.randn(*shape2)
    result = (Tensor.from_numpy(a) @ Tensor.from_numpy(b)).to_numpy()

    assert np.allclose(a @ b, result, atol=1e-6, rtol=1e-6)


def test_matmul_forward_2():
    shape1 = (batch, n, m)
    shape2 = (m, p)
    a = np.random.randn(*shape1)
    b = np.random.randn(*shape2)
    result = (Tensor.from_numpy(a) @ Tensor.from_numpy(b)).to_numpy()

    assert np.allclose(a @ b, result, atol=1e-6, rtol=1e-6)


def test_matmul_backward_1():
    shape1 = (n, m)
    shape2 = (m, p)

    # torch implementation
    a1 = torch.randn(*shape1, requires_grad=True)
    b1 = torch.randn(*shape2, requires_grad=True)
    c1 = a1 @ b1
    grad1 = torch.ones_like(c1)
    c1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = Tensor.from_torch(b1, requires_grad=True)

    c2 = a2 @ b2
    grad2 = Tensor.from_torch(grad1)
    c2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())
    assert torch.allclose(b1.grad, b2.get_grad().to_torch())


def test_matmul_backward_2():
    shape1 = (batch, n, m)
    shape2 = (m, p)

    # torch implementation
    a1 = torch.randn(*shape1, requires_grad=True)
    b1 = torch.randn(*shape2, requires_grad=True)
    c1 = a1 @ b1
    grad1 = torch.ones_like(c1)
    c1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = Tensor.from_torch(b1, requires_grad=True)

    c2 = a2 @ b2
    grad2 = Tensor.from_torch(grad1)
    c2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())
    assert torch.allclose(b1.grad, b2.get_grad().to_torch())

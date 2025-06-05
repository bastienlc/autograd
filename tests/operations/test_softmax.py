import numpy as np
import torch

from autograd import Tensor

np.random.seed(42)

n = 5


def test_softmax_forward():
    shape = (n,)
    a = np.random.randn(*shape)
    result = (Tensor.from_numpy(a).softmax()).to_numpy()

    assert np.allclose(np.exp(a - np.max(a)) / np.exp(a - np.max(a)).sum(), result)


def test_softmax_backward():
    shape = (n,)

    # torch implementation
    a1 = torch.randn(*shape, requires_grad=True)
    b1 = a1.softmax(0)
    grad1 = torch.rand_like(b1)  # 1-grad is unstable for the softmax formula
    b1.backward(grad1)

    # autograd implementation
    a2 = Tensor.from_torch(a1, requires_grad=True)
    b2 = a2.softmax()
    grad2 = Tensor.from_torch(grad1)
    b2.backward(grad2)

    # Check gradients
    assert torch.allclose(a1.grad, a2.get_grad().to_torch())

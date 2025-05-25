import random

import numpy as np

from autograd import Tensor

random.seed(42)

n = 10
m = 20
p = 30


def test_add():
    a = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a + b

    assert np.allclose(
        np.array(c.get_data()), np.array(a.get_data()) + np.array(b.get_data())
    )
    assert c.get_requires_grad() is True
    assert c.get_shape() == [n]


def test_mul():
    a = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a * b

    assert np.allclose(
        np.array(c.get_data()), np.array(a.get_data()) * np.array(b.get_data())
    )
    assert c.get_requires_grad() is True
    assert c.get_shape() == [n]


def test_matmul():
    a = Tensor(
        [n, m],
        np.array([random.random() for _ in range(n * m)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [m, p],
        np.array([random.random() for _ in range(m * p)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a @ b

    assert c.get_shape() == [n, p]
    assert c.get_requires_grad() is True
    assert np.allclose(
        np.array(c.get_data()),
        np.matmul(
            np.array(a.get_data()).reshape(a.get_shape()),
            np.array(b.get_data()).reshape(b.get_shape()),
        ).reshape(-1),
    )


def test_add_backward():
    a = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a + b
    c.reduce_sum().backward(None)

    assert a.get_grad() is not None
    assert b.get_grad() is not None
    assert c.get_grad() is not None
    assert np.allclose(a.get_grad().get_data(), np.ones(n))
    assert np.allclose(b.get_grad().get_data(), np.ones(n))
    assert np.allclose(c.get_grad().get_data(), np.ones(n))


def test_mul_backward():
    a = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [n],
        np.array([random.random() for _ in range(n)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a * b
    c.reduce_sum().backward(None)

    assert a.get_grad() is not None
    assert b.get_grad() is not None
    assert c.get_grad() is not None
    assert np.allclose(a.get_grad().get_data(), b.get_data())
    assert np.allclose(b.get_grad().get_data(), a.get_data())
    assert np.allclose(c.get_grad().get_data(), np.ones(n))


def test_matmul_backward():
    a = Tensor(
        [n, m],
        np.array([random.random() for _ in range(n * m)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    b = Tensor(
        [m, p],
        np.array([random.random() for _ in range(m * p)]),
        requires_grad=True,
        grad=None,
        graph=None,
    )
    c = a @ b
    c.reduce_sum().backward(None)

    assert a.get_grad() is not None
    assert b.get_grad() is not None
    assert c.get_grad() is not None
    assert c.get_shape() == [n, p]
    assert a.get_grad().get_shape() == [n, m]
    assert b.get_grad().get_shape() == [m, p]
    assert np.allclose(
        a.get_grad().get_data(),
        np.matmul(
            np.ones((n, p)),
            np.array(b.get_data()).reshape(b.get_shape()).transpose(),
        ).reshape(-1),
    )
    assert np.allclose(
        b.get_grad().get_data(),
        np.matmul(
            np.array(a.get_data()).reshape(a.get_shape()).transpose(),
            np.ones((n, p)),
        ).reshape(-1),
    )
    assert np.allclose(c.get_grad().get_data(), np.ones(n * p))


if __name__ == "__main__":
    test_add()
    test_mul()
    test_matmul()
    test_add_backward()
    test_mul_backward()
    test_matmul_backward()

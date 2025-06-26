import time

import matplotlib.pyplot as plt
from torch import Tensor as TorchTensor
from torch import ones_like, randn

from autograd import Tensor as AutogradTensor

K = 100
SHAPES = [2, 5, 10, 20, 50, 100, 200, 500]
UNARY_OPS = [
    ("neg", TorchTensor.__neg__, AutogradTensor.__neg__),
    ("relu", TorchTensor.relu, AutogradTensor.relu),
    ("softmax", lambda self: TorchTensor.softmax(self, 1), AutogradTensor.softmax),
    ("sum", TorchTensor.sum, AutogradTensor.reduce_sum),
    (
        "transpose",
        lambda self: TorchTensor.transpose(self, 0, 1),
        AutogradTensor.transpose,
    ),
]
BINARY_OPS = [
    ("add", TorchTensor.__add__, AutogradTensor.__add__),
    ("matmul", TorchTensor.__matmul__, AutogradTensor.__matmul__),
    ("mul", TorchTensor.__mul__, AutogradTensor.__mul__),
    ("sub", TorchTensor.__sub__, AutogradTensor.__sub__),
]


def do_profile():
    results = {}
    for n in SHAPES:
        print(n)
        for name, torch_op, autograd_op in BINARY_OPS:
            results[n, f"{name}_torch_forward"] = 0
            results[n, f"{name}_torch_backward"] = 0
            for _ in range(K):
                a, b = (
                    randn((n, n), requires_grad=True),
                    randn((n, n), requires_grad=True),
                )

                t = time.time()
                res = torch_op(a, b)
                results[n, f"{name}_torch_forward"] += time.time() - t

                grad = ones_like(res)
                t = time.time()
                res.backward(grad)
                results[n, f"{name}_torch_backward"] += time.time() - t

            results[n, f"{name}_autograd_forward"] = 0
            results[n, f"{name}_autograd_backward"] = 0
            for _ in range(K):
                a, b = (
                    AutogradTensor.from_torch(randn((n, n)), requires_grad=True),
                    AutogradTensor.from_torch(randn((n, n)), requires_grad=True),
                )
                t = time.time()
                res = autograd_op(a, b)
                results[n, f"{name}_autograd_forward"] += time.time() - t

                grad = AutogradTensor.from_torch(ones_like(res.to_torch()))
                t = time.time()
                res.backward(grad)
                results[n, f"{name}_autograd_backward"] += time.time() - t

        for name, torch_op, autograd_op in UNARY_OPS:
            results[n, f"{name}_torch_forward"] = 0
            results[n, f"{name}_torch_backward"] = 0
            for _ in range(K):
                a = randn((n, n), requires_grad=True)

                t = time.time()
                res = torch_op(a)
                results[n, f"{name}_torch_forward"] += time.time() - t

                grad = ones_like(res)
                t = time.time()
                res.backward(grad)
                results[n, f"{name}_torch_backward"] += time.time() - t

            results[n, f"{name}_autograd_forward"] = 0
            results[n, f"{name}_autograd_backward"] = 0
            for _ in range(K):
                a = AutogradTensor.from_torch(randn((n, n)), requires_grad=True)
                t = time.time()
                res = autograd_op(a)
                results[n, f"{name}_autograd_forward"] += time.time() - t

                grad = AutogradTensor.from_torch(ones_like(res.to_torch()))
                t = time.time()
                res.backward(grad)
                results[n, f"{name}_autograd_backward"] += time.time() - t
    return results


def plot_results(results):
    cols = 3
    rows = (len(UNARY_OPS) + len(BINARY_OPS) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for k, (op_name, _, _) in enumerate(UNARY_OPS + BINARY_OPS):
        x = SHAPES
        y_torch_f = [results[k, f"{op_name}_torch_forward"] for k in SHAPES]
        y_autograd_f = [results[k, f"{op_name}_autograd_forward"] for k in SHAPES]
        y_torch_b = [results[k, f"{op_name}_torch_backward"] for k in SHAPES]
        y_autograd_b = [results[k, f"{op_name}_autograd_backward"] for k in SHAPES]

        ax = axes[k]
        ax.plot(x, y_torch_f, label="Torch forward", marker="o", color="g")
        ax.plot(x, y_torch_b, label="Torch backward", marker="x", color="g")
        ax.plot(x, y_autograd_f, label="Autograd forward", marker="o", color="b")
        ax.plot(x, y_autograd_b, label="Autograd backward", marker="x", color="b")
        ax.set_title(op_name)
        ax.set_xlabel("N")
        ax.set_ylabel("Run time")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig("profiling.png")


if __name__ == "__main__":
    plot_results(do_profile())

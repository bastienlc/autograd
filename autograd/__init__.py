from typing import List

import numpy
import torch

from .autograd import Graph, Tensor

""" Useful methods for the autograd module."""


def tensor(shape: List[int], value: float, requires_grad: bool = False) -> Tensor:
    """
    Create a Tensor with the given shape and value.

    Args:
        shape (List[int]): The shape of the Tensor.
        value (float): The value to fill the Tensor with.
        requires_grad (bool): Whether the Tensor requires gradient computation.

    Returns:
        Tensor: The created Tensor.
    """
    return Tensor(
        shape=shape,
        data=[value] * numpy.prod(shape),
        requires_grad=requires_grad,
        grad=None,
        graph=None,
    )


""" We add some python methods to our objects here."""


def from_numpy(cls, np_array: numpy.ndarray, requires_grad: bool = False) -> Tensor:
    return cls(
        shape=list(np_array.shape),
        data=np_array.flatten().tolist(),
        requires_grad=requires_grad,
        grad=None,
        graph=None,
    )


def from_torch(cls, torch_tensor: torch.Tensor, requires_grad: bool = False) -> Tensor:
    return cls(
        shape=list(torch_tensor.shape),
        data=torch_tensor.flatten().tolist(),
        requires_grad=requires_grad,
        grad=None,
        graph=None,
    )


def to_numpy(self: Tensor) -> numpy.ndarray:
    return numpy.array(self.get_data()).reshape(self.get_shape())


def to_torch(self: Tensor) -> torch.Tensor:
    return torch.tensor(self.get_data()).reshape(self.get_shape())


Tensor.from_numpy = classmethod(from_numpy)  # type: ignore
Tensor.from_torch = classmethod(from_torch)  # type: ignore
Tensor.to_numpy = to_numpy  # type: ignore
Tensor.to_torch = to_torch  # type: ignore

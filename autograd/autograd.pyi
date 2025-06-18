from typing import List, Optional

import numpy
import torch

class Tensor:
    # Rust-defined methods

    def __new__(
        cls,
        shape: List[int],
        data: List[float],
        requires_grad: bool,
        grad: Optional[List[float]],
        graph: Optional[Graph],
    ): ...
    def get_shape(self) -> List[int]: ...
    def get_data(self) -> List[float]: ...
    def get_requires_grad(self) -> bool: ...
    def get_grad(self) -> Optional[Tensor]: ...
    def set_grad(self, grad: Optional[Tensor]) -> None: ...
    def get_graph(self) -> Optional[Graph]: ...
    def set_graph(self, graph: Optional[Graph]) -> None: ...
    def backward(self, grad: Optional[Tensor]) -> None: ...
    def __add__(self, other: Tensor) -> Tensor: ...
    def __sub__(self, other: Tensor) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...
    def transpose(self) -> Tensor: ...
    def reduce_sum(self) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def softmax(self) -> Tensor: ...
    def broadcast(self, shape: List[int]) -> Tensor: ...

    # Python-defined methods

    @classmethod
    def from_numpy(
        cls, np_array: numpy.ndarray, requires_grad: bool = False
    ) -> Tensor: ...
    """
    Convert a numpy array to a Tensor. The underlying data will be copied.

    Args:
        np_array (numpy.ndarray): The numpy array to convert.

    Returns:
        Tensor: The converted Tensor.
    """

    @classmethod
    def from_torch(
        cls, torch_tensor: torch.Tensor, requires_grad: bool = False
    ) -> Tensor: ...
    """
    Convert a torch.Tensor to a Tensor. The underlying data will be copied.

    Args:
        torch_tensor (torch.tensor): The torch tensor to convert.

    Returns:
        Tensor: The converted Tensor.
    """

    def to_numpy(self) -> numpy.ndarray: ...
    """
    Convert the Tensor to a numpy array.

    Returns:
        numpy.ndarray: The numpy array representation of the Tensor.
    """

    def to_torch(self) -> torch.Tensor: ...
    """
    Convert the Tensor to a torch.Tensor.

    Returns:
        torch.Tensor: The torch tensor representation of the Tensor.
    """

class Graph: ...

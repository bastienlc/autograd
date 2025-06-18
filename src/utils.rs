use std::sync::{Arc, Mutex};

use crate::{
    backward::Backward,
    objects::{Graph, Tensor},
    operations::broadcast::broadcast,
    DTYPE,
};

pub fn new_graph<T: Backward + Send + Sync + 'static>(op: T) -> Graph {
    Graph(Arc::new(Mutex::new(op)))
}

pub fn new_tensor_with_graph<T: Backward + Send + Sync + 'static>(
    shape: Vec<usize>,
    data: Vec<DTYPE>,
    requires_grad: bool,
    node: T,
) -> Tensor {
    Tensor::new(shape, data, requires_grad, None, Some(new_graph(node)))
}

pub fn new_tensor_simple(shape: Vec<usize>, data: Vec<DTYPE>) -> Tensor {
    Tensor::new(shape, data, false, None, None)
}

pub fn broadcast_to_same_dim(lhs: Tensor, rhs: Tensor) -> (Tensor, Tensor) {
    if lhs.get_shape() != rhs.get_shape() {
        if lhs.get_shape() == vec![1] {
            return (broadcast(lhs.clone(), rhs.get_shape()), rhs);
        } else if rhs.get_shape() == vec![1] {
            return (lhs.clone(), broadcast(rhs.clone(), lhs.get_shape()));
        } else if lhs.get_shape().len() == rhs.get_shape().len() + 1 {
            return (lhs.clone(), broadcast(rhs.clone(), lhs.get_shape()));
        } else if rhs.get_shape().len() == lhs.get_shape().len() + 1 {
            return (broadcast(lhs.clone(), rhs.get_shape()), rhs.clone());
        } else {
            panic!("broadcast_to_same_dim could not broadcast tensors with different shapes: {:?} and {:?}", lhs.get_shape(), rhs.get_shape());
        }
    } else {
        return (lhs, rhs);
    }
}

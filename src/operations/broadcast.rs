use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;

pub fn broadcast(t: Tensor, shape: Vec<usize>) -> Tensor {
    if t.get_shape() != vec![1] {
        panic!("Broadcasting is only defined for tensors with shape [1]");
    }

    return Tensor::new(
        shape.clone(),
        vec![t.get_data()[0]; shape.iter().product()],
        t.get_requires_grad(),
        None,
        Some(Graph::Broadcast(t)),
    );
}

#[pymethods]
impl Tensor {
    pub fn broadcast(&self, shape: Vec<usize>) -> Tensor {
        broadcast(self.clone(), shape)
    }
}

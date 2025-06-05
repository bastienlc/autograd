use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;

pub fn reduce_sum(t: Tensor) -> Tensor {
    let mut sum = 0.0;
    for i in t.get_data().iter() {
        sum += i;
    }
    return Tensor::new(
        vec![1],
        vec![sum],
        t.get_requires_grad(),
        None,
        Some(Graph::ReduceSum(t.clone())),
    );
}

#[pymethods]
impl Tensor {
    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self.clone())
    }
}

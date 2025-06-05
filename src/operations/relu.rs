use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;

pub fn relu(t: Tensor) -> Tensor {
    let data: Vec<f64> = t.get_data().iter().map(|&x| x.max(0.0)).collect();
    return Tensor::new(
        t.get_shape().clone(),
        data,
        t.get_requires_grad(),
        None,
        Some(Graph::Relu(t.clone())),
    );
}

#[pymethods]
impl Tensor {
    pub fn relu(&self) -> Tensor {
        relu(self.clone())
    }
}

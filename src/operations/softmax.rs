use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;

pub fn softmax(t: Tensor) -> Tensor {
    let max_val = t
        .get_data()
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_data: Vec<f64> = t.get_data().iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_data.iter().sum();
    let softmax_data: Vec<f64> = exp_data.iter().map(|&x| x / sum_exp).collect();

    return Tensor::new(
        t.get_shape().clone(),
        softmax_data,
        t.get_requires_grad(),
        None,
        Some(Graph::Softmax(t.clone())),
    );
}

#[pymethods]
impl Tensor {
    pub fn softmax(&self) -> Tensor {
        softmax(self.clone())
    }
}

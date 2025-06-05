use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;
use std::ops::Neg;

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let data: Vec<f64> = self.get_data().iter().map(|&x| -x).collect();
        return Tensor::new(
            self.get_shape(),
            data,
            self.get_requires_grad(),
            None,
            Some(Graph::Neg(self.clone())),
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn __neg__(&self) -> Tensor {
        -self.clone()
    }
}

use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;

pub fn transpose(t: Tensor) -> Tensor {
    let shape = t.get_shape();
    if shape.len() != 2 {
        panic!("Transpose is only defined for 2D tensors");
    }
    let new_shape = vec![shape[1], shape[0]];
    let mut data = vec![0.0; shape[0] * shape[1]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            data[j * shape[0] + i] = t.get_data()[i * shape[1] + j];
        }
    }
    return Tensor::new(
        new_shape,
        data,
        t.get_requires_grad(),
        None,
        Some(Graph::Transpose(t.clone())),
    );
}

#[pymethods]
impl Tensor {
    pub fn transpose(&self) -> Tensor {
        transpose(self.clone())
    }
}

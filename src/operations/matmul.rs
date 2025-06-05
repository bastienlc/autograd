use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    if a.get_shape().len() != 2 || b.get_shape().len() != 2 {
        panic!("Matrix multiplication requires both tensors to be 2D");
    }
    if a.get_shape()[1] != b.get_shape()[0] {
        panic!("Inner dimensions must match for matrix multiplication");
    }

    let m = a.get_shape()[0];
    let n = a.get_shape()[1];
    let p = b.get_shape()[1];

    let mut data = vec![0.0; m * p];

    data.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
        let a_row = &a.get_data()[i * n..i * n + n];
        for k in 0..n {
            let a_ik = a_row[k];
            let b_row = &b.get_data()[k * p..k * p + p];
            for j in 0..p {
                row[j] += a_ik * b_row[j];
            }
        }
    });

    return Tensor::new(
        vec![m, p],
        data,
        a.get_requires_grad() || b.get_requires_grad(),
        None,
        Some(Graph::MatMul(a.clone(), b.clone())),
    );
}

#[pymethods]
impl Tensor {
    pub fn __matmul__(&self, other: Tensor) -> Tensor {
        matmul(self.clone(), other)
    }
}

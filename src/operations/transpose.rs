use crate::{backward::Backward, objects::Tensor, utils::new_tensor_with_graph};
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
    return new_tensor_with_graph(
        new_shape,
        data,
        t.get_requires_grad(),
        TransposeOperation { t: t.clone() },
    );
}

pub struct TransposeOperation {
    t: Tensor,
}

impl Backward for TransposeOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        self.t.do_backward(Some(grad.transpose()), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn transpose(&self) -> Tensor {
        transpose(self.clone())
    }
}

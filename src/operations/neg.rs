use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
};
use pyo3::prelude::*;
use std::ops::Neg;

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let data: Vec<f64> = self.get_data().iter().map(|&x| -x).collect();
        return new_tensor_with_graph(
            self.get_shape(),
            data,
            self.get_requires_grad(),
            NegOperation { t: self.clone() },
        );
    }
}

pub struct NegOperation {
    t: Tensor,
}

impl Backward for NegOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let neg_grad = -grad.unwrap();
        self.t.do_backward(
            Some(new_tensor_simple(self.t.get_shape(), neg_grad.get_data())),
            None,
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn __neg__(&self) -> Tensor {
        -self.clone()
    }
}

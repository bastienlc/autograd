use crate::{backward::Backward, objects::Tensor, utils::new_tensor_with_graph, DTYPE};
use pyo3::prelude::*;
use std::ops::Neg;

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        let data: Vec<DTYPE> = self.get_data_ref().iter().map(|&x| -x).collect();
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
        self.t.do_backward(Some(neg_grad), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn __neg__(&self) -> Tensor {
        -self.clone()
    }
}

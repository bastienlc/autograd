use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{broadcast_to_same_dim, new_tensor_with_graph},
    DTYPE,
};
use pyo3::prelude::*;
use std::ops::Mul;

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        let (lhs, rhs) = broadcast_to_same_dim(self, rhs);

        let data: Vec<DTYPE> = lhs
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(a, b)| a * b)
            .collect();

        return new_tensor_with_graph(
            lhs.get_shape(),
            data,
            lhs.get_requires_grad() || rhs.get_requires_grad(),
            MulOperation {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            },
        );
    }
}

pub struct MulOperation {
    lhs: Tensor,
    rhs: Tensor,
}

impl Backward for MulOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        self.lhs
            .do_backward(Some(grad.clone() * self.rhs.clone()), None);
        self.rhs
            .do_backward(Some(grad.clone() * self.lhs.clone()), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn __mul__(&self, other: Tensor) -> Tensor {
        self.clone() * other
    }
}

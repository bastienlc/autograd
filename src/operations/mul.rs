use crate::{
    objects::{Graph, Tensor},
    operations::broadcast::broadcast,
};
use pyo3::prelude::*;
use std::ops::Mul;

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        if self.get_shape() != rhs.get_shape() {
            if self.get_shape() == vec![1] {
                return broadcast(self, rhs.get_shape()) + rhs;
            } else if rhs.get_shape() == vec![1] {
                return self.clone() + broadcast(rhs, self.get_shape());
            } else {
                panic!("Operation Mul requires tensors to have the same shape");
            }
        }
        let data: Vec<f64> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(a, b)| a * b)
            .collect();
        return Tensor::new(
            self.get_shape().clone(),
            data,
            self.get_requires_grad() || rhs.get_requires_grad(),
            None,
            Some(Graph::Mul(self.clone(), rhs.clone())),
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn __mul__(&self, other: Tensor) -> Tensor {
        self.clone() * other
    }
}

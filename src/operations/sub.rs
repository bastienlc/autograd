use crate::objects::Tensor;
use pyo3::prelude::*;
use std::ops::Sub;

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        return self + (-rhs);
    }
}

#[pymethods]
impl Tensor {
    pub fn __sub__(&self, other: Tensor) -> Tensor {
        self.clone() - other
    }
}

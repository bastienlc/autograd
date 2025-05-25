use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;
use std::ops::{Add, Mul};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        if self.get_shape() != rhs.get_shape() {
            panic!("Operation Add requires tensors to have the same shape");
        }
        let data: Vec<f64> = self
            .get_data()
            .iter()
            .zip(rhs.get_data().iter())
            .map(|(a, b)| a + b)
            .collect();
        return Tensor::new(
            self.get_shape().clone(),
            data,
            self.get_requires_grad() || rhs.get_requires_grad(),
            None,
            Some(Graph::Sum(
                Tensor {
                    core: self.core.clone(),
                },
                Tensor {
                    core: rhs.core.clone(),
                },
            )),
        );
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        if self.get_shape() != rhs.get_shape() {
            panic!("Operation Mul requires tensors to have the same shape");
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
            Some(Graph::Mul(
                Tensor {
                    core: self.core.clone(),
                },
                Tensor {
                    core: rhs.core.clone(),
                },
            )),
        );
    }
}

pub fn reduce_sum(t: Tensor) -> Tensor {
    let mut sum = 0.0;
    for i in t.get_data().iter() {
        sum += i;
    }
    return Tensor::new(
        vec![1],
        vec![sum],
        t.get_requires_grad(),
        None,
        Some(Graph::ReduceSum(Tensor {
            core: t.core.clone(),
        })),
    );
}

#[pymethods]
impl Tensor {
    pub fn __add__(&self, other: Tensor) -> Tensor {
        self.clone() + other
    }

    pub fn __mul__(&self, other: Tensor) -> Tensor {
        self.clone() * other
    }

    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self.clone())
    }
}

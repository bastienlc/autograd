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

    for i in 0..m {
        for j in 0..p {
            for k in 0..n {
                data[i * p + j] += a.get_data()[i * n + k] * b.get_data()[k * p + j];
            }
        }
    }

    return Tensor::new(
        vec![m, p],
        data,
        a.get_requires_grad() || b.get_requires_grad(),
        None,
        Some(Graph::MatMul(
            Tensor {
                core: a.core.clone(),
            },
            Tensor {
                core: b.core.clone(),
            },
        )),
    );
}

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
        Some(Graph::Transpose(Tensor {
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

    pub fn __matmul__(&self, other: Tensor) -> Tensor {
        matmul(self.clone(), other)
    }

    pub fn transpose(&self) -> Tensor {
        transpose(self.clone())
    }

    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self.clone())
    }
}

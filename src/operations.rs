use crate::objects::{Graph, Tensor};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        if self.get_shape() != rhs.get_shape() {
            if self.get_shape() == vec![1] {
                return broadcast(self, rhs.get_shape()) + rhs;
            } else if rhs.get_shape() == vec![1] {
                return self.clone() + broadcast(rhs, self.get_shape());
            } else {
                panic!("Operation Add requires tensors to have the same shape");
            }
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
            Some(Graph::Add(self.clone(), rhs.clone())),
        );
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Tensor {
        return self + neg(rhs);
    }
}

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

pub fn neg(t: Tensor) -> Tensor {
    let data: Vec<f64> = t.get_data().iter().map(|&x| -x).collect();
    return Tensor::new(
        t.get_shape(),
        data,
        t.get_requires_grad(),
        None,
        Some(Graph::Neg(t.clone())),
    );
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
        Some(Graph::ReduceSum(t.clone())),
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

pub fn relu(t: Tensor) -> Tensor {
    let data: Vec<f64> = t.get_data().iter().map(|&x| x.max(0.0)).collect();
    return Tensor::new(
        t.get_shape().clone(),
        data,
        t.get_requires_grad(),
        None,
        Some(Graph::Relu(t.clone())),
    );
}

pub fn softmax(t: Tensor) -> Tensor {
    let max_val = t
        .get_data()
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_data: Vec<f64> = t.get_data().iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_data.iter().sum();
    let softmax_data: Vec<f64> = exp_data.iter().map(|&x| x / sum_exp).collect();

    return Tensor::new(
        t.get_shape().clone(),
        softmax_data,
        t.get_requires_grad(),
        None,
        Some(Graph::Relu(t.clone())),
    );
}

pub fn broadcast(t: Tensor, shape: Vec<usize>) -> Tensor {
    if t.get_shape() != vec![1] {
        panic!("Broadcasting is only defined for tensors with shape [1]");
    }

    return Tensor::new(
        shape.clone(),
        vec![t.get_data()[0]; shape.iter().product()],
        t.get_requires_grad(),
        None,
        Some(Graph::Broadcast(t)),
    );
}

#[pymethods]
impl Tensor {
    pub fn __add__(&self, other: Tensor) -> Tensor {
        self.clone() + other
    }

    pub fn __sub__(&self, other: Tensor) -> Tensor {
        self.clone() - other
    }

    pub fn __neg__(&self) -> Tensor {
        neg(self.clone())
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

    pub fn relu(&self) -> Tensor {
        relu(self.clone())
    }

    pub fn softmax(&self) -> Tensor {
        softmax(self.clone())
    }

    pub fn broadcast(&self, shape: Vec<usize>) -> Tensor {
        broadcast(self.clone(), shape)
    }
}

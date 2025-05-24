use crate::objects::{Graph, Tensor};
use std::ops::{Add, Mul};

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        if self.shape() != rhs.shape() {
            panic!("Operation Add requires tensors to have the same shape");
        }
        let data: Vec<f64> = self
            .data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(a, b)| a + b)
            .collect();
        return Tensor::new(
            self.shape().clone(),
            data,
            self.requires_grad() || rhs.requires_grad(),
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
        if self.shape() != rhs.shape() {
            panic!("Operation Mul requires tensors to have the same shape");
        }
        let data: Vec<f64> = self
            .data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(a, b)| a * b)
            .collect();
        return Tensor::new(
            self.shape().clone(),
            data,
            self.requires_grad() || rhs.requires_grad(),
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
    for i in t.data().iter() {
        sum += i;
    }
    return Tensor::new(
        vec![1],
        vec![sum],
        t.requires_grad(),
        None,
        Some(Graph::ReduceSum(Tensor {
            core: t.core.clone(),
        })),
    );
}

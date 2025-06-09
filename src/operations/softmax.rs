use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
};
use pyo3::prelude::*;

pub fn softmax(t: Tensor) -> Tensor {
    let max_val = t
        .get_data()
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_data: Vec<f64> = t.get_data().iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_data.iter().sum();
    let softmax_data: Vec<f64> = exp_data.iter().map(|&x| x / sum_exp).collect();

    return new_tensor_with_graph(
        t.get_shape(),
        softmax_data,
        t.get_requires_grad(),
        SoftmaxOperation { t: t.clone() },
    );
}

pub struct SoftmaxOperation {
    t: Tensor,
}

impl Backward for SoftmaxOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, input: Option<Tensor>) {
        let grad = grad.unwrap();
        let input = input.unwrap();
        let prod_sum = input
            .get_data()
            .iter()
            .zip(grad.get_data().iter())
            .map(|(x, g)| x * g)
            .sum::<f64>();
        let softmax_grad = input
            .get_data()
            .iter()
            .zip(grad.get_data().iter())
            .map(|(x, g)| x * (g - prod_sum))
            .collect::<Vec<f64>>();
        self.t.do_backward(
            Some(new_tensor_simple(self.t.get_shape(), softmax_grad)),
            None,
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn softmax(&self) -> Tensor {
        softmax(self.clone())
    }
}

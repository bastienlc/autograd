use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
    DTYPE,
};
use pyo3::prelude::*;

pub fn softmax(t: Tensor) -> Tensor {
    let shape = t.get_shape();
    let data = t.get_data_ref();
    let dim = shape.last().copied().unwrap_or(1);
    let outer = data.len() / dim;

    let mut softmax_data = Vec::with_capacity(data.len());

    for i in 0..outer {
        let start = i * dim;
        let end = start + dim;
        let slice = &data[start..end];

        let max_val = slice.iter().cloned().fold(DTYPE::NEG_INFINITY, DTYPE::max);
        let exp_slice: Vec<DTYPE> = slice.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: DTYPE = exp_slice.iter().sum();
        softmax_data.extend(exp_slice.iter().map(|&x| x / sum_exp));
    }

    new_tensor_with_graph(
        shape,
        softmax_data,
        t.get_requires_grad(),
        SoftmaxOperation { t: t.clone() },
    )
}

pub struct SoftmaxOperation {
    t: Tensor,
}

impl Backward for SoftmaxOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, input: Option<Tensor>) {
        let grad = grad.unwrap();
        let input = input.unwrap();
        let shape = input.get_shape();
        let dim = shape.last().copied().unwrap_or(1);
        let outer = input.get_data_ref().len() / dim;

        let mut softmax_grad = Vec::with_capacity(input.get_data_ref().len());

        let input_data = input.get_data_ref();
        let grad_data = grad.get_data_ref();

        for i in 0..outer {
            let start = i * dim;
            let end = start + dim;
            let input_slice = &input_data[start..end];
            let grad_slice = &grad_data[start..end];

            let prod_sum: DTYPE = input_slice
                .iter()
                .zip(grad_slice.iter())
                .map(|(x, g)| x * g)
                .sum();

            softmax_grad.extend(
                input_slice
                    .iter()
                    .zip(grad_slice.iter())
                    .map(|(x, g)| x * (g - prod_sum)),
            );
        }

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

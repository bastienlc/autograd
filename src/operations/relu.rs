use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
    DTYPE,
};
use pyo3::prelude::*;

pub fn relu(t: Tensor) -> Tensor {
    let data: Vec<DTYPE> = t.get_data_ref().iter().map(|&x| x.max(0.0)).collect();
    return new_tensor_with_graph(
        t.get_shape(),
        data,
        t.get_requires_grad(),
        ReluOperation { t: t.clone() },
    );
}

pub struct ReluOperation {
    t: Tensor,
}

impl Backward for ReluOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        let relu_grad = self
            .t
            .get_data_ref()
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .zip(grad.get_data_ref().iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<DTYPE>>();
        self.t
            .do_backward(Some(new_tensor_simple(self.t.get_shape(), relu_grad)), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn relu(&self) -> Tensor {
        relu(self.clone())
    }
}

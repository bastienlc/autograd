use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
};
use pyo3::prelude::*;

pub fn reduce_sum(t: Tensor) -> Tensor {
    let mut sum = 0.0;
    for i in t.get_data().iter() {
        sum += i;
    }

    return new_tensor_with_graph(
        vec![1],
        vec![sum],
        t.get_requires_grad(),
        ReduceSumOperation { t: t.clone() },
    );
}

pub struct ReduceSumOperation {
    t: Tensor,
}

impl Backward for ReduceSumOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        self.t.do_backward(
            Some(new_tensor_simple(
                self.t.get_shape(),
                vec![grad.get_data()[0]; self.t.get_data().len()],
            )),
            None,
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn reduce_sum(&self) -> Tensor {
        reduce_sum(self.clone())
    }
}

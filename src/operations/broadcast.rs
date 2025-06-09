use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
};
use pyo3::prelude::*;

pub fn broadcast(t: Tensor, shape: Vec<usize>) -> Tensor {
    if t.get_shape() != vec![1] {
        panic!("Broadcasting is only defined for tensors with shape [1]");
    }

    return new_tensor_with_graph(
        shape.clone(),
        vec![t.get_data()[0]; shape.iter().product()],
        t.get_requires_grad(),
        BroadcastOperation { t: t.clone() },
    );
}

pub struct BroadcastOperation {
    t: Tensor,
}

impl Backward for BroadcastOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        return self.t.do_backward(
            Some(new_tensor_simple(
                self.t.get_shape(),
                grad.reduce_sum().get_data(),
            )),
            None,
        );
    }
}

#[pymethods]
impl Tensor {
    pub fn broadcast(&self, shape: Vec<usize>) -> Tensor {
        broadcast(self.clone(), shape)
    }
}

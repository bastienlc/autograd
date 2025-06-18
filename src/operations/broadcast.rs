use crate::{
    backward::Backward,
    objects::Tensor,
    utils::{new_tensor_simple, new_tensor_with_graph},
};
use pyo3::prelude::*;

pub fn broadcast(t: Tensor, shape: Vec<usize>) -> Tensor {
    if t.get_shape() == vec![1] {
        return new_tensor_with_graph(
            shape.clone(),
            vec![t.get_data()[0]; shape.iter().product()],
            t.get_requires_grad(),
            BroadcastOperation {
                t: t.clone(),
                shape: shape,
            },
        );
    }

    if shape.len() != t.get_shape().len() + 1 {
        panic!(
            "Cannot broadcast tensor with shape {:?} to shape {:?}",
            t.get_shape(),
            shape
        );
    }

    for (dim1, dim2) in t.get_shape().iter().zip(shape[1..].iter()) {
        if *dim1 != *dim2 {
            panic!(
                "Cannot broadcast tensor with shape {:?} to shape {:?}",
                t.get_shape(),
                shape
            );
        }
    }

    let batch_size = shape[0];
    return new_tensor_with_graph(
        shape.clone(),
        t.get_data().repeat(batch_size),
        t.get_requires_grad(),
        BroadcastOperation {
            t: t.clone(),
            shape: shape,
        },
    );
}

pub struct BroadcastOperation {
    t: Tensor,
    shape: Vec<usize>,
}

impl Backward for BroadcastOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();

        if self.t.get_shape() == vec![1] {
            return self.t.do_backward(
                Some(new_tensor_simple(
                    self.t.get_shape(),
                    grad.reduce_sum().get_data(),
                )),
                None,
            );
        } else {
            let mut summed_grad = vec![0.0; self.t.get_data().len()];
            for i in 0..self.shape[0] {
                for j in 0..self.t.get_data().len() {
                    summed_grad[j] += grad.get_data()[i * self.t.get_data().len() + j];
                }
            }
            return self.t.do_backward(
                Some(new_tensor_simple(self.t.get_shape(), summed_grad)),
                None,
            );
        }
    }
}

#[pymethods]
impl Tensor {
    pub fn broadcast(&self, shape: Vec<usize>) -> Tensor {
        broadcast(self.clone(), shape)
    }
}

use crate::{objects::Tensor, utils::new_tensor_simple};
use pyo3::prelude::*;

pub trait Backward {
    fn do_backward(&mut self, grad: Option<Tensor>, input: Option<Tensor>);
}

impl Backward for Tensor {
    /// Backward pass for the tensor.
    /// If the tensor is a scalar, it will create a gradient of 1.
    /// Otherwise, it will panic if no gradient is provided.
    fn do_backward(&mut self, grad: Option<Tensor>, input: Option<Tensor>) {
        if !self.get_requires_grad() {
            return;
        }
        if self.get_grad().is_none() {
            // TODO: without this we can deadlock.
            // self.core.write().unwrap().grad should be hidden within the tensor API
            let length = self.get_data_ref().len();
            /* Initializing the gradient to zero if it is None */
            self.core.write().unwrap().grad =
                Some(new_tensor_simple(self.get_shape(), vec![0.0; length]));
        }
        if grad.is_none() {
            /* grad can be None if the tensor is a scalar */
            if self.get_shape().len() > 1 || self.get_shape()[0] != 1 {
                panic!("Backward requires grad to be provided for non-scalar tensors");
            } else {
                self.core.write().unwrap().grad =
                    Some(Tensor::new(vec![1], vec![1.0], false, None, None));
            }
        } else {
            /* Accumulating the gradient */
            let current_grad = self.core.read().unwrap().grad.clone().unwrap();
            self.core.write().unwrap().grad = Some(grad.unwrap() + current_grad);
        }
        if !input.is_none() {
            panic!("Expected input to be None for Tensor backward");
        }

        /* Go up the graph */
        match self.get_graph() {
            None => {
                return;
            }
            Some(ref mut graph) => {
                graph
                    .0
                    .write()
                    .unwrap()
                    .do_backward(self.get_grad(), Some(self.clone()));
            }
        }
    }
}

#[pymethods]
impl Tensor {
    pub fn backward(&mut self, grad: Option<Tensor>) {
        self.do_backward(grad, None);
    }
}

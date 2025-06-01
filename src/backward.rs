use crate::{
    objects::{Graph, Tensor},
    operations::matmul,
};
use pyo3::prelude::*;

pub trait Backward {
    fn do_backward(&mut self, grad: Option<Tensor>);
}

impl Backward for Tensor {
    /// Backward pass for the tensor.
    /// If the tensor is a scalar, it will create a gradient of 1.
    /// Otherwise, it will panic if no gradient is provided.
    fn do_backward(&mut self, grad: Option<Tensor>) {
        if !self.get_requires_grad() {
            return;
        }
        if grad.is_none() {
            /* grad can be None if the tensor is a scalar */
            if self.get_shape().len() > 1 || self.get_shape()[0] != 1 {
                panic!("Backward requires grad to be provided for non-scalar tensors");
            } else {
                self.core.lock().unwrap().grad =
                    Some(Tensor::new(vec![1], vec![1.0], false, None, None));
            }
        } else {
            self.core.lock().unwrap().grad = grad;
        }

        /* Go up the graph */
        match self.get_graph() {
            None => {
                return;
            }
            Some(ref mut graph) => {
                graph.do_backward(self.get_grad());
            }
        }
    }
}

impl Backward for Graph {
    fn do_backward(&mut self, grad: Option<Tensor>) {
        if grad.is_none() {
            panic!("Expected grad to be provided for Graph backward");
        }

        let grad = grad.unwrap();

        match self {
            // x = y + z
            // dx/dy = 1 and dx/dz = 1
            // dy = grad and dz = grad
            Graph::Add(left, right) => {
                left.do_backward(Some(grad.clone()));
                right.do_backward(Some(grad.clone()));
            }
            // x = -y
            // dx/dy = -1
            // dy = -grad
            Graph::Neg(t) => {
                let neg_grad = t
                    .get_data()
                    .iter()
                    .map(|&x| -x)
                    .zip(grad.get_data().iter())
                    .map(|(a, b)| a * b)
                    .collect::<Vec<f64>>();
                t.do_backward(Some(Tensor::new(
                    t.get_shape(),
                    neg_grad,
                    false,
                    None,
                    None,
                )));
            }
            // x = y * z
            // dx/dy = z and dx/dz = y
            // dy = grad * z and dz = grad * y
            Graph::Mul(left, right) => {
                left.do_backward(Some(grad.clone() * right.clone()));
                right.do_backward(Some(grad.clone() * left.clone()));
            }
            // x = y @ z
            // dx/dy = z^T and dx/dz = y^T
            // dy = grad @ z^T and dz = y^T @ grad
            Graph::MatMul(left, right) => {
                left.do_backward(Some(matmul(grad.clone(), right.transpose())));
                right.do_backward(Some(matmul(left.transpose(), grad)));
            }
            // x = y^T
            // dx/dy = 1
            // dy = grad^T
            Graph::Transpose(t) => {
                t.do_backward(Some(grad.transpose()));
            }
            // x = reduce_sum(y)
            // dx/dy = 1 for all elements in y
            // dy = grad * [1, 1, ..., 1]
            Graph::ReduceSum(t) => {
                if grad.get_shape().len() > 1 || grad.get_shape()[0] != 1 {
                    panic!("ReduceSum operation should leave a scalar gradient");
                }
                let shape = t.get_shape();
                let grad_size = t.get_data().len();
                t.do_backward(Some(Tensor::new(
                    shape,
                    vec![grad.get_data()[0]; grad_size],
                    false,
                    None,
                    None,
                )));
            }
            // x = relu(y)
            // if input > 0; then grad; else 0
            Graph::Relu(t) => {
                let relu_grad = t
                    .get_data()
                    .iter()
                    .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                    .zip(grad.get_data().iter())
                    .map(|(a, b)| a * b)
                    .collect::<Vec<f64>>();
                t.do_backward(Some(Tensor::new(
                    t.get_shape(),
                    relu_grad,
                    false,
                    None,
                    None,
                )));
            }
            // x = softmax(y)
            // softmax(y) * (grad - sum(softmax(y) * grad))
            Graph::Softmax(t) => {
                let softmax_data = t.get_data();
                let sum_softmax_grad: f64 = softmax_data
                    .iter()
                    .zip(grad.get_data().iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let softmax_grad = softmax_data
                    .iter()
                    .zip(grad.get_data().iter())
                    .map(|(a, b)| a * (b - sum_softmax_grad))
                    .collect::<Vec<f64>>();
                t.do_backward(Some(Tensor::new(
                    t.get_shape(),
                    softmax_grad,
                    false,
                    None,
                    None,
                )));
            }
            // x = broadcast(y)
            // sum the gradient across the broadcasted dimensions (all)
            Graph::Broadcast(t) => {
                if t.get_shape().len() != 1 || t.get_shape()[0] != 1 {
                    panic!("Broadcast operation should have shape [1]");
                }
                return t.do_backward(Some(Tensor::new(
                    t.get_shape(),
                    grad.reduce_sum().get_data(),
                    false,
                    None,
                    None,
                )));
            }
        }
    }
}

#[pymethods]
impl Tensor {
    pub fn backward(&mut self, grad: Option<Tensor>) {
        self.do_backward(grad);
    }
}

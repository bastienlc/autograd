use crate::objects::{Graph, Tensor};

pub trait Backward {
    fn backward(&mut self, grad: Option<Tensor>);
}

impl Backward for Tensor {
    /// Backward pass for the tensor.
    /// If the tensor is a scalar, it will create a gradient of 1.
    /// Otherwise, it will panic if no gradient is provided.
    fn backward(&mut self, grad: Option<Tensor>) {
        if !self.requires_grad() {
            return;
        }
        if grad.is_none() {
            /* grad can be None if the tensor is a scalar */
            if self.shape().len() > 1 || self.shape()[0] != 1 {
                panic!("Backward requires grad to be provided for non-scalar tensors");
            } else {
                self.core.borrow_mut().grad =
                    Some(Tensor::new(vec![1], vec![1.0], false, None, None));
            }
        } else {
            self.core.borrow_mut().grad = grad;
        }

        /* Go up the graph */
        match self.graph() {
            None => {
                return;
            }
            Some(ref mut graph) => {
                graph.backward(self.grad());
            }
        }
    }
}

impl Backward for Graph {
    fn backward(&mut self, grad: Option<Tensor>) {
        if grad.is_none() {
            panic!("Expected grad to be provided for Graph backward");
        }

        let grad = grad.unwrap();

        match self {
            // x = y + z
            // dx/dy = 1 and dx/dz = 1
            // dy = grad and dz = grad
            Graph::Sum(left, right) => {
                left.backward(Some(grad.clone()));
                right.backward(Some(grad.clone()));
            }
            // x = y * z
            // dx/dy = z and dx/dz = y
            // dy = grad * z and dz = grad * y
            Graph::Mul(left, right) => {
                left.backward(Some(grad.clone() * right.clone()));
                right.backward(Some(grad.clone() * left.clone()));
            }
            // x = reduce_sum(y)
            // dx/dy = 1 for all elements in y
            // dy = grad * [1, 1, ..., 1]
            Graph::ReduceSum(t) => {
                if grad.shape().len() > 1 || grad.shape()[0] != 1 {
                    panic!("REDUCE_SUM operation should leave a scalar gradient");
                }
                let shape = t.shape();
                let grad_size = t.data().len();
                t.backward(Some(Tensor::new(
                    shape,
                    vec![grad.data()[0]; grad_size],
                    false,
                    None,
                    None,
                )));
            }
        }
    }
}

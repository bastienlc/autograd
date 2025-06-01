use crate::objects::{Graph, Tensor};
use std::fmt;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Tensor {{ shape: {:?}, data: {:?} }}",
            self.get_shape(),
            self.get_data()
        )
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Graph::Add(left, right) => write!(f, "SUM({}, {})", left, right),
            Graph::Neg(t) => write!(f, "NEG({})", t),
            Graph::Mul(left, right) => write!(f, "MUL({}, {})", left, right),
            Graph::MatMul(left, right) => write!(f, "MATMUL({}, {})", left, right),
            Graph::Transpose(t) => write!(f, "TRANSPOSE({})", t),
            Graph::ReduceSum(t) => write!(f, "REDUCE_SUM({})", t),
            Graph::Relu(t) => write!(f, "RELU({})", t),
            Graph::Softmax(t) => write!(f, "SOFTMAX({})", t),
            Graph::Broadcast(t) => write!(f, "BROADCAST({})", t),
        }
    }
}

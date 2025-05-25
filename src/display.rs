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
            Graph::Sum(left, right) => write!(f, "SUM({}, {})", left, right),
            Graph::Mul(left, right) => write!(f, "MUL({}, {})", left, right),
            Graph::ReduceSum(t) => write!(f, "REDUCE_SUM({})", t),
        }
    }
}

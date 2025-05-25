use pyo3::prelude::*;

use std::{cell::RefCell, rc::Rc};

/* Tensor is the main object we manipulate */

#[derive(Clone)]
pub struct CoreTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub graph: Option<Graph>,
}

#[pyclass(unsendable)] // TODO should make type thread-safe (Send + Sync)
#[derive(Clone)]
pub struct Tensor {
    pub core: Rc<RefCell<CoreTensor>>,
}
#[pymethods]
impl Tensor {
    #[new]
    pub fn new(
        shape: Vec<usize>,
        data: Vec<f64>,
        requires_grad: bool,
        grad: Option<Tensor>,
        graph: Option<Graph>,
    ) -> Self {
        if shape.iter().product::<usize>() != data.len() {
            panic!("Shape and data length do not match");
        }
        Tensor {
            core: Rc::new(RefCell::new(CoreTensor {
                shape,
                data,
                requires_grad,
                grad: grad,
                graph: graph,
            })),
        }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.core.borrow().shape.clone()
    }
    pub fn get_data(&self) -> Vec<f64> {
        self.core.borrow().data.clone()
    }
    pub fn get_requires_grad(&self) -> bool {
        self.core.borrow().requires_grad
    }
    pub fn get_grad(&self) -> Option<Tensor> {
        self.core.borrow().grad.clone()
    }
    pub fn get_graph(&self) -> Option<Graph> {
        self.core.borrow().graph.clone()
    }
}

#[pyfunction]
pub fn strides(t: Tensor) -> Vec<usize> {
    let shape = t.get_shape();
    let mut strides = vec![1];
    for i in (0..shape.len() - 1).rev() {
        strides.push(strides.last().unwrap() * shape[i + 1]);
    }
    strides.reverse();
    strides
}

/* Holds the computation graph of the tensor */

#[pyclass(unsendable)] // TODO should make type thread-safe (Send + Sync)
#[derive(Clone)]
pub enum Graph {
    Sum(Tensor, Tensor),
    Mul(Tensor, Tensor),
    ReduceSum(Tensor),
}

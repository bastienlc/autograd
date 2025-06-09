use pyo3::prelude::*;

use std::sync::{Arc, Mutex};

use crate::backward::Backward;

/* Tensor is the main object we manipulate */
pub struct CoreTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub graph: Option<Graph>,
}

#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    pub core: Arc<Mutex<CoreTensor>>,
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
            core: Arc::new(Mutex::new(CoreTensor {
                shape,
                data,
                requires_grad,
                grad: grad,
                graph: graph,
            })),
        }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.core.lock().unwrap().shape.clone()
    }
    pub fn get_data(&self) -> Vec<f64> {
        self.core.lock().unwrap().data.clone()
    }
    pub fn get_requires_grad(&self) -> bool {
        self.core.lock().unwrap().requires_grad
    }
    pub fn get_grad(&self) -> Option<Tensor> {
        self.core.lock().unwrap().grad.clone()
    }
    pub fn get_graph(&self) -> Option<Graph> {
        self.core.lock().unwrap().graph.clone()
    }
    pub fn set_graph(&mut self, graph: Option<Graph>) {
        self.core.lock().unwrap().graph = graph;
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

#[pyclass]
#[derive(Clone)]
pub struct Graph(pub Arc<Mutex<dyn Backward + Send + Sync>>);

use pyo3::prelude::*;

use std::sync::{Arc, RwLock};

use crate::{backward::Backward, DTYPE};

/* Tensor is the main object we manipulate */
pub struct CoreTensor {
    pub shape: Vec<usize>,
    pub data: Vec<DTYPE>,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub graph: Option<Graph>,
}

#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    pub core: Arc<RwLock<CoreTensor>>,
}
#[pymethods]
impl Tensor {
    #[new]
    pub fn new(
        shape: Vec<usize>,
        data: Vec<DTYPE>,
        requires_grad: bool,
        grad: Option<Tensor>,
        graph: Option<Graph>,
    ) -> Self {
        if shape.iter().product::<usize>() != data.len() {
            panic!("Shape and data length do not match");
        }
        Tensor {
            core: Arc::new(RwLock::new(CoreTensor {
                shape,
                data,
                requires_grad,
                grad: grad,
                graph: graph,
            })),
        }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.core.read().unwrap().shape.clone()
    }
    pub fn get_data(&self) -> Vec<DTYPE> {
        self.core.read().unwrap().data.clone()
    }
    pub fn get_requires_grad(&self) -> bool {
        self.core.read().unwrap().requires_grad
    }
    pub fn get_grad(&self) -> Option<Tensor> {
        self.core.read().unwrap().grad.clone()
    }
    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        self.core.write().unwrap().grad = grad;
    }
    pub fn get_graph(&self) -> Option<Graph> {
        self.core.read().unwrap().graph.clone()
    }
    pub fn set_graph(&mut self, graph: Option<Graph>) {
        self.core.write().unwrap().graph = graph;
    }
}

// These methods are not safe to expose to Python
impl Tensor {
    pub fn get_data_ref(&'_ self) -> std::sync::MappedRwLockReadGuard<'_, Vec<DTYPE>> {
        let core = self.core.read().unwrap();
        std::sync::RwLockReadGuard::map(core, |core| &core.data)
    }
}

pub fn strides(shape: &Vec<usize>) -> Vec<usize> {
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
pub struct Graph(pub Arc<RwLock<dyn Backward + Send + Sync>>);

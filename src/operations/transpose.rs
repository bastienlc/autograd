use crate::{
    backward::Backward,
    objects::{strides, Tensor},
    utils::new_tensor_with_graph,
};
use pyo3::prelude::*;

pub fn transpose(t: Tensor) -> Tensor {
    let shape = t.get_shape();
    if shape.len() < 2 {
        panic!("Transpose is only defined for tensors with at least 2 dimensions");
    }
    let new_shape = {
        let mut new_shape = shape.clone();
        new_shape[shape.len() - 1] = shape[shape.len() - 2];
        new_shape[shape.len() - 2] = shape[shape.len() - 1];
        new_shape
    };
    let size = new_shape.iter().product();
    let ndim = new_shape.len();

    let mut data = vec![0.0; size];
    let old_strides = strides(&shape);
    let new_strides = strides(&new_shape);

    // iterate over the old shape indices
    for i in 0..size {
        // compute ndim index in old shape
        let mut old_index = vec![0; ndim];
        for j in 0..ndim {
            old_index[j] = (i / old_strides[j]) % shape[j];
        }
        let mut new_index = old_index.clone();
        // swap the last two indices
        new_index[ndim - 1] = old_index[ndim - 2];
        new_index[ndim - 2] = old_index[ndim - 1];
        // compute linear index for the new shape
        let mut new_linear_index = 0;
        for j in 0..ndim {
            new_linear_index += new_index[j] * new_strides[j];
        }
        // assign the value
        data[new_linear_index] = t.get_data_ref()[i];
    }

    return new_tensor_with_graph(
        new_shape,
        data,
        t.get_requires_grad(),
        TransposeOperation { t: t.clone() },
    );
}

pub struct TransposeOperation {
    t: Tensor,
}

impl Backward for TransposeOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        self.t.do_backward(Some(grad.transpose()), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn transpose(&self) -> Tensor {
        transpose(self.clone())
    }
}

use crate::{
    backward::Backward, objects::Tensor, operations::broadcast::broadcast,
    utils::new_tensor_with_graph, DTYPE,
};
use pyo3::prelude::*;
use rayon::prelude::*;

pub fn matul_kernel(
    lhs: &Vec<DTYPE>,
    rhs: &Vec<DTYPE>,
    m: usize,
    n: usize,
    p: usize,
) -> Vec<DTYPE> {
    let mut data = vec![0.0; m * p];
    data.par_chunks_mut(p).enumerate().for_each(|(i, row)| {
        let a_row = &lhs[i * n..i * n + n];
        for k in 0..n {
            let a_ik = a_row[k];
            let b_row = &rhs[k * p..k * p + p];
            for j in 0..p {
                row[j] += a_ik * b_row[j];
            }
        }
    });
    return data;
}

pub fn batch_matmul_kernel(
    lhs: &Vec<DTYPE>,
    rhs: &Vec<DTYPE>,
    m: usize,
    n: usize,
    p: usize,
    batch_size: usize,
) -> Vec<DTYPE> {
    let mut data = vec![0.0; batch_size * m * p];
    data.par_chunks_mut(m * p)
        .enumerate()
        .for_each(|(b, chunk)| {
            let a_batch = &lhs[b * m * n..(b + 1) * m * n];
            let b_batch = &rhs[b * n * p..(b + 1) * n * p];
            chunk.copy_from_slice(&matul_kernel(&a_batch.to_vec(), &b_batch.to_vec(), m, n, p));
        });
    return data;
}

pub fn matmul(lhs: Tensor, rhs: Tensor) -> Tensor {
    let m: usize;
    let p: usize;
    let shape: Vec<usize>;
    let data: Vec<DTYPE>;

    if lhs.get_shape().len() == 2 && rhs.get_shape().len() == 2 {
        m = lhs.get_shape()[0];
        p = rhs.get_shape()[1];
        let n = lhs.get_shape()[1];
        if n != rhs.get_shape()[0] {
            panic!("Inner dimensions must match for matrix multiplication");
        }
        shape = vec![m, p];
        data = matul_kernel(&lhs.get_data_ref(), &rhs.get_data_ref(), m, n, p);
    } else if lhs.get_shape().len() == 3 && rhs.get_shape().len() == 3 {
        let batch_size = lhs.get_shape()[0];
        m = lhs.get_shape()[1];
        p = rhs.get_shape()[2];
        let n = lhs.get_shape()[2];
        if n != rhs.get_shape()[1] {
            panic!("Inner dimensions must match for batch matrix multiplication");
        }
        shape = vec![batch_size, m, p];
        data = batch_matmul_kernel(
            &lhs.get_data_ref(),
            &rhs.get_data_ref(),
            m,
            n,
            p,
            batch_size,
        );
    } else if lhs.get_shape().len() == 3 && rhs.get_shape().len() == 2 {
        let shape = vec![lhs.get_shape()[0], rhs.get_shape()[0], rhs.get_shape()[1]];
        return matmul(lhs, broadcast(rhs, shape));
    } else {
        panic!(
            "Matrix multiplication is only defined for 2D or 3D tensors. Got shapes: {:?} and {:?}",
            lhs.get_shape(),
            rhs.get_shape()
        );
    }

    return new_tensor_with_graph(
        shape,
        data,
        lhs.get_requires_grad() || rhs.get_requires_grad(),
        MatMulOperation {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        },
    );
}

pub struct MatMulOperation {
    lhs: Tensor,
    rhs: Tensor,
}

impl Backward for MatMulOperation {
    fn do_backward(&mut self, grad: Option<Tensor>, _: Option<Tensor>) {
        let grad = grad.unwrap();
        self.lhs
            .do_backward(Some(matmul(grad.clone(), self.rhs.transpose())), None);
        self.rhs
            .do_backward(Some(matmul(self.lhs.transpose(), grad)), None);
    }
}

#[pymethods]
impl Tensor {
    pub fn __matmul__(&self, other: Tensor) -> Tensor {
        matmul(self.clone(), other)
    }
}

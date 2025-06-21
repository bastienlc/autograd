#![feature(mapped_lock_guards)]

use pyo3::prelude::*;

pub mod backward;
pub mod eq;
pub mod objects;
pub mod operations;
pub mod utils;

#[pymodule]
fn autograd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<objects::Tensor>()?;
    m.add_class::<objects::Graph>()?;
    Ok(())
}

pub type DTYPE = f32;

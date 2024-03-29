#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

extern crate approx;
extern crate ndarray_linalg;

pub mod algebra;
pub mod eig;
pub mod quantum;
pub mod reps;
pub mod util;

pub use simba::scalar::{RealField, ComplexField, SubsetOf, SupersetOf, ClosedAdd, ClosedMul};

pub use nalgebra::{SimdComplexField, SimdRealField};

pub use util::scalar::{RealScalar, ComplexScalar};

pub use num_complex;

#[cfg(test)]
mod tests {
    extern crate openblas_src;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

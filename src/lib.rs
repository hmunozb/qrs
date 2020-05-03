#![allow(unused_imports)]
#![allow(unused_variables)]
/// QRS - A quantum library for rust
///
extern crate alga;
extern crate approx;
extern crate log;
extern crate nalgebra;

//extern crate packed_simd;
//pub use qrs_core::ComplexField;
pub use qrs_core::{ComplexField, RealField};
pub use qrs_core::{ComplexScalar, RealScalar};

pub mod util;
pub mod algebra;
pub mod base;
pub mod oqs;
pub mod ode;
pub mod semi;

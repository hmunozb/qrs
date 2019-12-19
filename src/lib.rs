///

#[macro_use]
extern crate approx;
extern crate alga;
extern crate nalgebra;
extern crate log;

use alga::general::{RealField, ComplexField};
use blas_traits::BlasScalar;
use num_complex::Complex32 as c32;
use num_complex::Complex64 as c64;

pub trait ComplexScalar<R: RealField> :
    ComplexField<RealField=R> + BlasScalar {}
impl ComplexScalar<f32> for f32{}
impl ComplexScalar<f64> for f64{}
impl ComplexScalar<f32> for num_complex::Complex<f32>{}
impl ComplexScalar<f64> for num_complex::Complex<f64>{}

pub mod util;
pub mod algebra;
pub mod base;
pub mod oqs;
pub mod ode;


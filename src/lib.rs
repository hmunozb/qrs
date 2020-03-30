///

extern crate approx;
extern crate alga;
extern crate nalgebra;
extern crate log;
//extern crate packed_simd;

use alga::general::{RealField, ComplexField};
use lapack_traits::LapackScalar;


pub trait ComplexScalar<R: RealField> :
    ComplexField<RealField=R> + LapackScalar {}
impl ComplexScalar<f32> for f32{}
impl ComplexScalar<f64> for f64{}
impl ComplexScalar<f32> for num_complex::Complex<f32>{}
impl ComplexScalar<f64> for num_complex::Complex<f64>{}

pub mod util;
pub mod algebra;
pub mod base;
pub mod oqs;
pub mod ode;
pub mod semi;

extern crate approx;
extern crate ndarray_linalg;

pub mod algebra;
pub mod eig;
pub mod quantum;
pub mod reps;
pub mod util;

pub use simba::scalar::{RealField, ComplexField, SubsetOf, SupersetOf, ClosedAdd, ClosedMul};
//use alga::general::{RealField, ComplexField};
use lapack_traits::LapackScalar;
use num_traits::{Float, NumOps, ToPrimitive};

pub use nalgebra::{SimdComplexField, SimdRealField};

pub trait RealScalar : RealField
+ vec_ode::RealField
+ LapackScalar
+ ndarray_linalg::Lapack
//+ alga::general::RealField
+ ndarray::ScalarOperand
+ ToPrimitive
//+ cauchy::Scalar<Real=Self>
//+ Float + NumOps
{ }
impl<R> RealScalar for R where R: RealField
+ vec_ode::RealField
+ LapackScalar
+ ndarray_linalg::Lapack
//+ alga::general::RealField
+ ToPrimitive
//+ cauchy::Scalar<Real=Self>
//+ Float + NumOps
+ ndarray::ScalarOperand,
// <Self as ndarray_linalg::Scalar>::Complex : NumOps<Self>
//+ cauchy::Scalar<Real=Self>
{ }

pub trait ComplexScalar :
ComplexField<RealField=<Self as ComplexScalar>::R>
//+ NumOps<R, Self> +
//+ cauchy::Scalar
+ LapackScalar
+ ndarray_linalg::Lapack
//+ alga::general::ComplexField<RealField=<Self as ComplexScalar>::R>
+ ndarray::ScalarOperand
//+ cauchy::Scalar<Real=<Self as ComplexScalar>::R>
//where <Self as cauchy::Scalar>::Complex : NumOps<R, Self>
//where <Self as cauchy::Scalar>::Real : RealField
{
    type R : RealScalar;
}

impl ComplexScalar for f32{ type R = f32;}
impl ComplexScalar for f64{ type R = f64;}
impl ComplexScalar for num_complex::Complex<f32>{type R = f32;}
impl ComplexScalar for num_complex::Complex<f64>{type R = f64;}

#[cfg(test)]
mod tests {
    extern crate openblas_src;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

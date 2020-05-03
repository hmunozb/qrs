
pub trait RealScalar :
 nalgebra::RealField +
 vec_ode::RealField
//+ LapackScalar
//+ ndarray_linalg::Lapack
//+ alga::general::RealField
//+ ndarray::ScalarOperand
//+ ToPrimitive
//+ cauchy::Scalar<Real=Self>
//+ Float + NumOps
{ }
impl<R> RealScalar for R where R:
  nalgebra::RealField
+ vec_ode::RealField
//+ LapackScalar
//+ ndarray_linalg::Lapack
//+ alga::general::RealField
//+ ToPrimitive
//+ cauchy::Scalar<Real=Self>
//+ Float + NumOps
//+ ndarray::ScalarOperand,
// <Self as ndarray_linalg::Scalar>::Complex : NumOps<Self>
//+ cauchy::Scalar<Real=Self>
{ }

pub trait ComplexScalar :
nalgebra::ComplexField<RealField=<Self as ComplexScalar>::R>
//+ NumOps<R, Self> +
//+ cauchy::Scalar
//+ LapackScalar
//+ ndarray_linalg::Lapack
//+ alga::general::ComplexField<RealField=<Self as ComplexScalar>::R>
//+ ndarray::ScalarOperand
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
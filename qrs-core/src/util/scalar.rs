/// RealScalar and ComplexScalar are the base traits that must be
/// implemented by the scalars of all QReps

pub trait RealScalar :
 nalgebra::RealField +
 vec_ode::RealField
{ }
impl<R> RealScalar for R where R:
  nalgebra::RealField
+ vec_ode::RealField
{ }

pub trait ComplexScalar :
nalgebra::ComplexField<RealField=<Self as ComplexScalar>::R>
{
    type R : RealScalar + Into<Self>;

    fn i() -> Self;
}

impl ComplexScalar for f32{
    type R = f32;
    fn i() -> f32{
        panic!("No imaginary unit for f32")
    }
}
impl ComplexScalar for f64{
    type R = f64;
    fn i() -> f64{
        panic!("No imaginary unit for f64")
    }
}
impl ComplexScalar for num_complex::Complex<f32>{
    type R = f32;
    fn i() -> num_complex::Complex<f32>{
        return num_complex::Complex::i();
    }
}
impl ComplexScalar for num_complex::Complex<f64>{
    type R = f64;
    fn i() -> num_complex::Complex<f64>{
        return num_complex::Complex::i();
    }
}
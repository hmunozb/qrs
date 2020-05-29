


use nalgebra::{Dim, Matrix, Vector};
use std::borrow::Cow;

pub struct Ket<'a, N: nalgebra::Scalar, R: Dim, S: Clone>(Cow<'a, Vector<N, R, S>>);
pub struct Op<'a, N: nalgebra::Scalar, R: Dim, C: Dim, S: Clone>(Cow<'a, Matrix<N, R, C ,S>>);


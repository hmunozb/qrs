use num_traits::real::Real;

use crate::ComplexScalar;
use crate::quantum::{QObj, QRep};

pub mod dense;
pub mod dmatrix;

#[derive(Copy,Clone)]
pub enum EigJob{
    ValsVecs,
    ValsOnly
}

#[derive(Copy,Clone)]
pub enum EigRange<F: Real>{
    All,
    ValRange(F, F),
    IdxRange(i32, i32)
}

/// A quantum representation where operators can be diagonalized
pub trait EigQRep<N: ComplexScalar> : QRep<N>{
    fn eig(op: &Self::OpRep) -> (Vec<N::R>, Self::OpRep );
}


/// A quantum representation where operators can be diagonalized
/// with a concrete instantiation of the representation, which may be required
/// to hold additional parameters
pub trait QEiger<N: ComplexScalar, Q> where Q: QRep<N>{
    fn make_eiger(shape: <Q::OpRep as QObj<N>>::Dims, job: EigJob, range: EigRange<N::R>) -> Self;
    fn eigh(&mut self, op: &Q::OpRep) -> (Vec<N::R>, Q::OpRep );
}
use sprs::{CsMat, CsVec};

use crate::quantum::*;
use std::marker::PhantomData;
use core::iter::Sum;

pub trait Scalar: crate::ComplexScalar + Sum + Default{ }

impl<N> Scalar for N where N: crate::ComplexScalar + Sum + Default { }

pub type Ket<N> = CsVec<N>;
pub type Bra<N> = ConjugatingWrapper<CsVec<N>>;
pub type Op<N> = CsMat<N>;

#[derive(Clone)]
pub struct SparseQRep<N>
where N: Scalar{
    _phantom: PhantomData<N>
}

pub struct LC<N> where N: Scalar{
    _phantom: PhantomData<N>
}

impl<N: Scalar> QRep<N> for SparseQRep<N>{
    type KetRep = Ket<N>;
    type BraRep = Bra<N>;
    type OpRep = Op<N>;

    fn qbdot(u: &Self::BraRep, v: &Self::KetRep) -> N {
        use sprs::vec::SparseIterTools;
        u.q.dot(v);
        v.iter().nnz_zip(u.q.iter())
            .map(|(_,&vi, &ui)| vi * ui)
            .sum()
    }

    fn qdot(u: &Ket<N>, v: &Ket<N>) -> N {
        use sprs::vec::SparseIterTools;

        u.iter()
         .nnz_zip(
        v.iter()  )
        .map(|(_,&ui, &vi)| ui.conjugate() * vi)
        .sum()

    }

    fn khemv(op: &Op<N>, alpha: N, x: &Ket<N>, y: &mut Self::KetRep, beta: N) {

        let mut op_x  = op * x;
        op_x.map_inplace(|&xi| alpha * xi);
        y.map_inplace(|&yi| beta * yi);
        let b  = &*y + op_x;
        *y = b;

    }
}

impl<N: Scalar> QObj<N> for Ket<N>{
    type Rep = SparseQRep<N>;
    type Dims = usize;

    fn qdim(&self) -> Self::Dims {
        self.dim()
    }

    fn qtype(&self) -> QType {
        QType::QKet
    }

    fn qaxpy(&mut self, a: N, x: &Self) {
        let mut ax = x.clone();
        ax.qscal(a);
        *self = &*self + ax;
    }

    fn qscal(&mut self, a: N) {
        self.map_inplace(|&xi| a*xi);
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        self.map_inplace(|&yi| a*yi);
        QObj::qaxpy(self, a, x);
    }
}


impl<N: Scalar> QObj<N> for Op<N>{
    type Rep = SparseQRep<N>;
    type Dims = sprs::Shape;

    fn qdim(&self) -> Self::Dims {
        self.shape()
    }

    fn qtype(&self) -> QType {
        QType::QOp(QOpType::Ge)
    }

    fn qaxpy(&mut self, a: N, x: &Self) {
        let mut ax  = x.clone();
        ax.qscal(a);
        let b = &*self + &ax;
        *self = b;
    }

    fn qscal(&mut self, a: N) {
        self.map_inplace(|&xi| a*xi);
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        self.map_inplace(|&yi| a*yi);
        QObj::qaxpy(self, a, x);
    }
}
impl<N: Scalar> QKet<N> for Ket<N>{

}

impl<N: Scalar> QOp<N> for Op<N>{

}

impl<N: Scalar> QBra<N> for Bra<N>{

}
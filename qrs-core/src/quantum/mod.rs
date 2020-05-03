use std::ops::{AddAssign, SubAssign, Index, IndexMut};
use crate::ComplexScalar;
pub mod eig;

/// Type of QObj
/// Necessary to check not known at compile time, e.g. functions accepting a dyn QObj
#[derive(Copy, Clone)]
pub enum QType{
    QKet,
    QBra,
    QOp(QOpType),
    QSop,
    QObj,
    QScal
}

#[derive(Copy, Clone)]
pub enum QOpType{
    He, Un, Ge
}

/// Trait implementing all common operations of a quantum object
/// In general, this includes any linear operations
pub trait QObj<N: ComplexScalar> :
Sized + Clone +
    //for<'b> AddAssign<&'b Self> +
    //for<'b> SubAssign<&'b Self>
{
    type Rep: QRep<N>;
    type Dims: Sized;

    /// The dimensions of the quantum object
    fn qdim(&self) -> Self::Dims;

    /// Specific type of quantum object
    fn qtype(&self) -> QType;

    /// y <- ax + y
    fn qaxpy(&mut self, a: N, x: &Self);

    /// x <- a x
    fn qscal(&mut self, a: N);

    /// y <- ax + by
    fn qaxby(&mut self, a: N, x: &Self, b: N);
}

/// Quantum Representation Trait
/// Implemented for zero-size/small structures to define how to perform
/// certain quantum operations in the given representation.
pub trait QRep <N: ComplexScalar>: Clone{
    type KetRep:    QKet<N, Rep=Self>;
    type BraRep:    QBra<N, Rep=Self>;
    type OpRep:     QOp<N, Rep=Self>;

    fn qdim_op(op: &Self::OpRep) -> <Self::OpRep as QObj<N>>::Dims {
        op.qdim()
    }
    fn qdim_ket(ket: &Self::KetRep) -> <Self::KetRep as QObj<N>>::Dims {
        ket.qdim()
    }
    fn qdim_bra(bra: &Self::BraRep) ->  <Self::BraRep as QObj<N>>::Dims {
        bra.qdim()
    }

    /// Takes the dot product of the bra and ket, where the bra is already in a conjugate representation
    fn qbdot(bra: &Self::BraRep, ket: &Self::KetRep) -> N;

    /// Takes the dot product of the conjugate of u with v
    fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> N;

    // /// Adjoint-Swap the bra and ket, i.e
    // ///   bra <- ket^H
    // ///   ket <- bra^H
    // fn qswap(bra: &mut Self::BraRep, ket: &mut Self::KetRep);

    /// ket <- alpha * op * ket_x + beta * ket_y
    fn khemv(op: &Self::OpRep, alpha:N, x: & Self::KetRep, y: &mut Self::KetRep, beta: N);

    /// Scales ket by the complex scalar a
    fn kscal(a: N, ket: &mut Self::KetRep){
        ket.qscal(a)
    }
    // fn qscal(a: N, op: &mut Self::OpRep){
    //     op.qscal(a)
    // }

    /// Scales ket by the real scalar a
    fn krscal(a: N::RealField, ket: &mut Self::KetRep){
        Self::kscal(N::from_real(a), ket);
    }

    /// Calculates y <- a x + y
    fn kaxpy(a: N, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep){
        ket_y.qaxpy(a, ket_x)
    }

    //fn eig(op: &Self::OpRep) -> (Vec<N::RealField>, Self::OpRep );
}


/// Trait for explicitly finite-dimensional quantum representations
///
pub trait FDimQRep<N: ComplexScalar, Arr>
: QRep<N>
    where
{
    type KetBasis;

    fn qstack(kets: Self::KetBasis) -> Self::OpRep;

}

pub trait QRepFunctor<N1, N2, Q1, Q2>
where   N1: ComplexScalar, Q1: QRep<N1>,
        N2: ComplexScalar, Q2: QRep<N2>
{
    fn map_ket(q1_ket: Q1::KetRep) -> Q2::KetRep;
    fn map_bra(q1_bra: Q1::BraRep) -> Q2::BraRep;
    fn map_op(q1_op: Q1::OpRep) -> Q2::OpRep;
}


pub trait TensorProd<N: ComplexScalar, RHS=Self> : QObj<N>{
    type Result: QObj<N>;

    fn tensor(a: Self, b: RHS) -> Self::Result;
    fn tensor_ref(a: &Self, b: &RHS) -> Self::Result;
}

pub trait QKet<N: ComplexScalar>: QObj<N>
    where Self::Rep : QRep<N, KetRep=Self>
{
    //type Rep: QRep<N, KetRep=Self>;

    //fn qdim(&self) -> usize;

    fn qdot(&self, other: & <<Self as QObj<N>>::Rep as QRep<N>>::BraRep) -> N{
        <Self::Rep as QRep<N>>::qbdot(other, self)
    }
    fn scal(&mut self, a: N) {
        <Self::Rep as QRep<N>>::kscal(a, self)
    }

}

pub trait QBra<N: ComplexScalar>: QObj<N>
    where Self::Rep : QRep<N, BraRep=Self>
{
    //type Rep: QRep<N, BraRep=Self>;

    fn qdot(&self, other: &<<Self as QObj<N>>::Rep as QRep<N>>::KetRep) -> N{
        <Self::Rep as QRep<N>>::qbdot(self, other)
    }

}


pub trait QOp<N: ComplexScalar> : QObj<N>
    where
        Self::Rep : QRep<N, OpRep=Self>
{
    //type Rep: QRep<N, OpRep=Self>;

    // fn qdim(&self) -> <Self as QObj<R,N>>::Dims{
    //     QObj::qdim(self)
    //     //<Self::Rep as QRep<R,N>>::qdim_op(self)
    // }
    // fn scal(&mut self, a: N) {
    //     <Self::Rep as QRep<R,N>>::qscal(a, self)
    // }
}

pub fn qdot<QB, QK, QR, N>(bra: &QB, ket: &QK) -> N
    where N: ComplexScalar,
          QR: QRep<N, KetRep=QK, BraRep=QB>,
          QB: QBra<N, Rep=QR>,
          QK: QKet<N, Rep=QR>
{
    bra.qdot(ket)
}


#[derive(Clone)]
pub struct ConjugatingWrapper<Q: Clone>{
    pub q: Q
}

impl<Q: Clone> From<Q> for ConjugatingWrapper<Q>{
    /// Construct directly from q as though it were conjugated
    fn from(q: Q) -> Self {
        Self{q}
    }
}


impl<N: ComplexScalar, Q> QObj<N> for ConjugatingWrapper<Q>
where Q: QObj<N>{
    type Rep = Q::Rep;
    type Dims = Q::Dims;

    fn qdim(&self) -> Self::Dims {
        self.q.qdim()
    }

    fn qtype(&self) -> QType {
        let qt = self.q.qtype();
        match qt{
            QType::QKet => QType::QBra,
            QType::QBra => QType::QKet,
            _ => qt
        }
    }

    fn qaxpy(&mut self, a: N, x: &Self) {
        self.q.qaxpy(a, &x.q);
    }

    fn qscal(&mut self, a: N) {
        self.q.qscal(a)
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        self.q.qaxby(a, &x.q, b)
    }
}
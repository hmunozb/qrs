use std::ops::{AddAssign, SubAssign};
//use alga::general::{ClosedAdd, ClosedSub};
use alga::general::ComplexField;
//use alga::linear::{VectorSpace, NormedSpace, InnerSpace};

//use num_traits::real::Real;
//use approx::AbsDiffEq;


pub trait QObj<N: ComplexField> :
//VectorSpace<Field=N> +
Sized +
for<'b> AddAssign<&'b Self> +
for<'b> SubAssign<&'b Self>{

}

/// Quantum Representation
pub trait QRep<N: ComplexField>: Sized{
    type KetRep:    QKet<N, Rep=Self>;
    type BraRep:    QBra<N, Rep=Self>;
    type OpRep:     QOp<N, Rep=Self>;

    fn qdim_op(op: &Self::OpRep) -> usize;
    fn qdim_ket(ket: &Self::KetRep) -> usize;
    fn qdim_bra(bra: &Self::BraRep) -> usize;

    /// Takes the dot product of the bra and ket
    fn qdot(bra: &Self::BraRep, ket: &Self::KetRep) -> N;

    /// Adjoint-Swap the bra and ket, i.e
    ///   bra <- ket^H
    ///   ket <- bra^H
    fn qswap(bra: &mut Self::BraRep, ket: &mut Self::KetRep);

    /// ket <- alpha * op * ket_x + beta * ket_y
    fn khemv(op: &Self::OpRep, alpha:N, x: & Self::KetRep, y: &mut Self::KetRep, beta: N);

    /// Scales ket by the complex scalar a
    fn kscal(a: N, ket: &mut Self::KetRep);
    fn qscal(a: N, op: &mut Self::OpRep);
    /// Scales ket by the real scalar a
    fn krscal(a: N::RealField, ket: &mut Self::KetRep){
        Self::kscal(N::from_real(a), ket);
    }

    /// Calculates y <- a x + y
    fn kaxpy(a: N, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep);

    fn eig(op: &Self::OpRep) -> (Vec<N::RealField>, Self::OpRep );
}

pub trait TensorProd<N: ComplexField, RHS=Self> : QObj<N>{
    type Result: QObj<N>;

    fn tensor(a: Self, b: RHS) -> Self::Result;
    fn tensor_ref(a: &Self, b: &RHS) -> Self::Result;
}

pub trait QKet<N: ComplexField>: QObj<N> {
    type Rep: QRep<N, KetRep=Self>;

    fn dim(&self) -> usize{
        <Self::Rep as QRep<N>>::qdim_ket(self)
    }

    fn qdot(&self, other: & <<Self as QKet<N>>::Rep as QRep<N>>::BraRep) -> N{
        <Self::Rep as QRep<N>>::qdot(other, self)
    }
    fn scal(&mut self, a: N) {
        <Self::Rep as QRep<N>>::kscal(a, self)
    }

}

pub trait QBra<N: ComplexField>: QObj<N>{
    type Rep: QRep<N, BraRep=Self>;

    fn qdot(&self, other: &<<Self as QBra<N>>::Rep as QRep<N>>::KetRep) -> N{
        <Self::Rep as QRep<N>>::qdot(self, other)
    }

}


pub trait QOp<N: ComplexField>: QObj<N>{
    type Rep: QRep<N, OpRep=Self>;


    fn qdim(&self) -> usize{
        <Self::Rep as QRep<N>>::qdim_op(self)
    }
    fn scal(&mut self, a: N) {
        <Self::Rep as QRep<N>>::qscal(a, self)
    }
}




//
//struct KetClass<N: ComplexField, D: Dim, R: QRep<N, D>> {
//    _ket: R::KetRep
//}
//
//struct BraClass<N: ComplexField, D: Dim, R: QRep<N, D>> {
//    _bra: Matrix<N, U1, D, R::BraStorage>
//}
//
//struct OpClass<N: ComplexField, D: Dim, R: QRep<N, D>> {
//    _op: Matrix<N, D, D, R::OpStorage>
//}
//
//impl<N: ComplexField, D: Dim, R: QRep<N, D>> AddAssign for KetClass<N, D, R>{
//    fn add_assign(&mut self, other: Self){
//        self._ket += other._ket;
//    }
//}
//impl<N: ComplexField, D: Dim, R: QRep<N, D>> AddAssign<&KetClass<N, D, R>> for KetClass<N, D, R>{
//    fn add_assign(&mut self, other: & Self){
//        //self._ket += other._ket;
//    }
//}

//impl<N: ComplexField, D: Dim, R: QRep<N, D>> QClass<N, D> for KetClass<N, D, R>
//where R::KetRep : AddAssign
//{
//    type QRep = R;
//
//
//}
//
//struct Qobj<N: ComplexField, D: Dim, Q: QClass<N, D>>{
//    _obj: Q
//}
//
//impl<N: ComplexField, D: Dim, Q: QClass<N, D>> Qobj<N, D, Q>{
//
//}
//
//impl<N: ComplexField, D: Dim, R: QRep<N, D>> Qobj<N, D, KetClass<N, D, R>>{
//
//}

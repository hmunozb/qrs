use std::marker::PhantomData;

//use num_complex::Complex;
use nalgebra::{DMatrix, DVector, Matrix};
use nalgebra::Dim;
use nalgebra::base::storage::StorageMut;
use vec_ode::LinearCombination;

use crate::quantum::NormedQRep;
use crate::ComplexScalar;
use crate::quantum::*;

pub type Ket<N> = DVector<N>;
pub type Bra<N> = DVector<N>;
pub type Op<N> =  DMatrix<N>;

#[derive(Clone)]
pub struct DenseQRep<N: ComplexScalar>
{ _phantom: PhantomData<N> }

#[derive(Clone)]
pub struct LC<N> where N: ComplexScalar{
    _phantom: PhantomData<N>
}

impl<N: ComplexScalar > QRep<N> for DenseQRep<N>
{
    type KetRep = Ket<N>;
    type BraRep = Bra<N>;
    type OpRep = Op<N>;


    fn qbdot(bra: &Self::BraRep, ket: & Self::KetRep) -> N{
        bra.dot(ket)
    }

    fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> N {
        u.dotc(v)
    }

    fn khemv(op: &Self::OpRep, alpha: N,
             x: & Self::KetRep, y: &mut Self::KetRep, beta: N){
        use nalgebra::Vector;
        Vector::gemv(y, alpha, op, x, beta)

    }

    fn kscal(a: N, ket: &mut Self::KetRep){
        ket.apply(|q| q*a);
    }

    fn kaxpy(a: N, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep){
        ket_y.zip_apply(ket_x, |y,x| a*x + y);
    }
}

impl<N: ComplexScalar > NormedQRep<N> for DenseQRep<N>{
    fn ket_norm_sq(v: &Self::KetRep) -> <N as ComplexScalar>::R {
        return v.norm_squared();
    }

    fn ket_norm(v: &Self::KetRep) -> <N as ComplexScalar>::R {
        return v.norm();
    }
}

impl<N, R, C, S> LinearCombination<N, Matrix<N, R, C, S>> for LC<N>
    where N: ComplexScalar,
          R: Dim, C: Dim, S: StorageMut<N, R, C>
{
    fn scale(v: &mut Matrix<N, R, C, S>, k: N) {
        *v *= k;
    }

    fn scalar_multiply_to(v: &Matrix<N, R, C, S>, k: N, target: &mut Matrix<N, R, C, S>) {
        target.zip_apply(v, |_t, vi| vi * k);
    }

    fn add_scalar_mul(v: &mut Matrix<N, R, C, S>, k: N, other: &Matrix<N, R, C, S>) {
        v.zip_apply(other, move |y, x| y + (k * x));
    }

    fn add_assign_ref(v: &mut Matrix<N, R, C, S>, other: &Matrix<N, R, C, S>) {
        *v += other;
    }

    fn delta(v: &mut Matrix<N, R, C, S>, y: &Matrix<N, R, C, S>) {
        *v -= y;
    }
}

impl<N: ComplexScalar, R: Dim, C: Dim, S: StorageMut<N, R, C>+Clone >
QObj<N>
for Matrix<N, R, C, S>
{
    type Rep = DenseQRep<N>;
    type Dims = (usize, usize);

    fn qdim(&self) -> Self::Dims {
        self.shape()
    }

    fn qtype(&self) -> QType {
        let sh = self.shape();
        match sh{
            (1, 1) => QType::QScal,
            (1, n) => QType::QKet,
            (n, 1) => QType::QBra,
            (n, m) =>{
                QType::QOp(QOpType::Ge)
            }
        }
        //return QType::QObj;
    }

    fn qaxpy(&mut self, a: N, x: &Self) {
         self.zip_apply(x, |yi, xi| a*xi +  yi);
    }

    fn qscal(&mut self, a: N) {
        self.apply(|x| a*x);
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        if b.is_zero(){
            self.zip_apply(x,|_yi, xi| a*xi);
        } else {
            self.zip_apply(x,  |yi, xi| a*xi + b*yi);
        }
    }
}

impl<N: ComplexScalar>
TensorProd<N, Op<N>>
for Op<N>
{
    type Result = Op<N>;

    fn tensor(a: Self, b:  Op<N>) -> Self::Result{
        a.kronecker(&b)
    }

    fn tensor_ref(a: &Self, b: & Op<N>)-> Self::Result{
        a.kronecker(&b)
    }
}

/// Evaluate the tensor product of the slice of operators
/// where the first operator contributes to the *least significant*
/// digit of the tensor space index. That is, if the argument is the 2D operators
/// `[s0, s1, ..., sn]`
/// then the tensor operator is
/// ```math
///     S = s_n \otimes \ldots \otimes s_1 \otimes s_0
/// ```
/// and the least significant bit of each index of S is the index for s_0
pub fn tensor_list<N: ComplexScalar>(ops: &[ Op<N>]) -> Op<N>{
    if ops.len() == 0{
        panic!("tensor_list Op slice cannot be empty");
    }
    let (first, rest) = ops.split_at(1);
    let mut v = first[0].clone();
    for u in rest.iter(){
        v = TensorProd::tensor_ref(u, &v)
    }

    v
}

impl<N: ComplexScalar>
TensorProd<N, Ket<N>>
for Ket<N>{
    type Result = Ket<N>;

    fn tensor(a: Self, b:  Ket<N>) -> Self::Result{
        a.kronecker(&b)
    }

    fn tensor_ref(a: &Self, b: & Ket<N>)-> Self::Result{
        a.kronecker(&b)
    }
}

//impl<N: RealField>
//TensorProd<Complex<N>, Bra<N>>
//for Bra<N>{
//    type Result = Bra<N>;
//
//    fn tensor(a: Self, b:  Bra<N>) -> Self::Result{
//        a.kronecker(&b)
//    }
//
//    fn tensor_ref(a: &Self, b: & Bra<N>)-> Self::Result{
//        a.kronecker(&b)
//    }
//}

impl<N: ComplexScalar> QOp<N> for Op<N>
{
    //type Rep = DenseQRep<N>;
}

impl<N: ComplexScalar> QKet<N> for Ket<N>
{
    //type Rep = DenseQRep<N>;
}

impl<N: ComplexScalar> QBra<N> for Bra<N>
{
    //type Rep = DenseQRep<N>;
}

#[cfg(test)]
mod tests{
    use crate::quantum::qdot;

    use super::{Bra, Ket, QBra};

    #[test]
    fn test_dense_qrep(){
        let a : Bra<f64> = Bra::zeros(3);
        let b : Ket<f64> = Ket::zeros(3);
        //b.qdot(a);
        let c = QBra::qdot(&a, &b);
        let d = qdot(&a, &b);
    }
}
use super::quantum::*;
use crate::util::{EigResolver, EigJob, EigRangeData};
use alga::general::{RealField, ComplexField};
use num_complex::Complex;
use nalgebra::{Matrix, DVector, DMatrix};
use nalgebra::{Dim};
use nalgebra::base::storage::{StorageMut};
use std::marker::PhantomData;
use crate::ComplexScalar;

pub type Ket<N> = DVector<Complex<N>>;
pub type Bra<N> = DVector<Complex<N>>;
pub type Op<N> =  DMatrix<Complex<N>>;

pub struct DenseQRep<N>
where N: ComplexField
{ _phantom: PhantomData<N> }

impl<R: RealField, > QRep<Complex<R>> for DenseQRep<R>
where Complex<R>: ComplexScalar<R>
{
    type KetRep = Ket<R>;
    type BraRep = Bra<R>;
    type OpRep = Op<R>;

    fn qdim_op(op: &Self::OpRep) -> usize {
        return op.nrows();
    }

    fn qdim_ket(ket: &Self::KetRep) -> usize {
        return ket.len();
    }

    fn qdim_bra(bra: &Self::BraRep) -> usize {
        return bra.len();
    }


    fn qbdot(bra: &Self::BraRep, ket: & Self::KetRep) -> Complex<R>{
        bra.dot(ket)
    }

    fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> Complex<R> {
        u.dotc(v)
    }

    fn qswap(bra: &mut Self::BraRep, ket: & mut Self::KetRep){
        for (q,r) in bra.as_mut_slice().iter_mut().zip(
            ket.as_mut_slice().iter_mut()){
            std::mem::swap(q, r);
        }
        bra.conjugate_mut();
        ket.conjugate_mut();
    }

    fn khemv(op: &Self::OpRep, _alpha: Complex<R>,
             x: & Self::KetRep, y: &mut Self::KetRep, _beta: Complex<R>){
        //if beta.eq(&N::zero()){
            op.mul_to(x, y);
       // } else {
        //    y.hegemv()
       // }
    }

    fn kscal(a: Complex<R>, ket: &mut Self::KetRep){
        ket.apply(|q| q*a);
    }

    fn qscal(a: Complex<R>, op: &mut Self::OpRep) {op.apply(|q| q*a);}

    fn kaxpy(a: Complex<R>, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep){
        ket_y.zip_apply(ket_x, |x,y| a*x + y);
    }

    fn eig(op: & Self::OpRep) -> (Vec<R>, Self::OpRep){
        let mut eiger:EigResolver<Complex<R>> = EigResolver::new_eiger(
            Self::qdim_op(op) as u32, EigJob::ValsVecs,
            EigRangeData::<R>::all()
        );
        let m = eiger.borrow_matrix();
        m.copy_from(op);
        let (vals, vecs) = eiger.into_eigs();
        let vals: Vec<R> = vals.data.into();

        (vals, vecs)
    }
}


impl<N: RealField, R: Dim, C: Dim, S: StorageMut<Complex<N>, R, C> >
QObj<Complex<N>>
for Matrix<Complex<N>, R, C, S>{ }

impl<N: RealField,
//    R1: Dim, C1: Dim, S1: StorageMut<N, R1, C1>,
//    R2: Dim, C2: Dim, S2: StorageMut<N, R2, C2>
>

TensorProd<Complex<N>, Op<N> >
for Op<N>{
    type Result = Op<N>;

    fn tensor(a: Self, b:  Op<N>) -> Self::Result{
        a.kronecker(&b)
    }

    fn tensor_ref(a: &Self, b: & Op<N>)-> Self::Result{
        a.kronecker(&b)
    }
}

pub fn tensor_list<N:RealField>(ops: &[ Op<N>]) -> Op<N>{
    if ops.len() == 0{
        panic!("tensor_list Op slice cannot be empty");
    }
    let (first, rest) = ops.split_at(1);
    let mut v = first[0].clone();
    for u in rest.iter(){
        v = TensorProd::tensor_ref(&v, &u)
    }

    v
}

impl<N: RealField>
TensorProd<Complex<N>, Ket<N>>
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

impl<N: RealField> QOp<Complex<N>> for Op<N>
where Complex<N> : ComplexScalar<N>
{
    type Rep = DenseQRep<N>;
}

impl<N: RealField> QKet<Complex<N>> for Ket<N>
where Complex<N>: ComplexScalar<N>
{
    type Rep = DenseQRep<N>;
}

impl<N: RealField> QBra<Complex<N>> for Bra<N>
where Complex<N>: ComplexScalar<N>
{
    type Rep = DenseQRep<N>;
}

#[cfg(test)]
mod tests{
    use super::{Bra, Ket, QKet, QBra};
    use crate::base::quantum::qdot;
    #[test]
    fn test_dense_qrep(){
        let a : Bra<f64> = Bra::zeros(3);
        let b : Ket<f64> = Ket::zeros(3);
        //b.qdot(a);
        let c = QBra::qdot(&a, &b);
        let d = qdot(&a, &b);
    }
}
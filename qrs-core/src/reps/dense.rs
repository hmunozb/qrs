use std::marker::PhantomData;

use ndarray::prelude::*;
use vec_ode::LinearCombination;

use crate::quantum::*;
use crate::util::array::kronecker;

//use ndarray_linalg::eigh::Eigh;

pub trait Scalar : ndarray_linalg::Scalar + crate::ComplexScalar
{ }
impl<N> Scalar for N where N: ndarray_linalg::Scalar + crate::ComplexScalar { }

pub type Ket<N> = Array1<N>;
pub type Bra<N> = ConjugatingWrapper<Array1<N>>;
pub type Op<N> =  Array2<N>;

#[derive(Clone)]
pub struct DenseQRep<N>
    where N: Scalar
{ _phantom: PhantomData<N> }

pub struct LC<N> where N: Scalar{
    _phantom: PhantomData<N>
}

impl<N: Scalar > QRep<N> for DenseQRep<N>
//where Complex<R>: Scalar<R>
{
    type KetRep = Ket<N>;
    type BraRep = Bra<N>;
    type OpRep = Op<N>;

    fn qdim_op(op: &Self::OpRep) -> (usize, usize) {
        let shape = op.shape();
        return (shape[0], shape[1])
    }

    fn qdim_ket(ket: &Self::KetRep) -> usize {
        return ket.len();
    }

    fn qdim_bra(bra: &Self::BraRep) -> usize {
        return bra.q.len();
    }


    fn qbdot(bra: &Self::BraRep, ket: & Self::KetRep) -> N{
        bra.q.dot(ket)
    }

    fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> N {
        u.iter().zip(v.iter())
            .map(|(&ui,&vi)| N::conj(&ui) * vi)
            .fold(N::zero(), |d, xi|d + xi) // sum not autoimpl
    }

    // fn qswap(bra: &mut Self::BraRep, ket: & mut Self::KetRep){
    //     for (q,r) in bra.q.iter_mut().zip(
    //         ket.iter_mut()){
    //         std::mem::swap(q, r);
    //     }
    //     bra.q.mapv_inplace(|x| x.conjugate());
    //     ket.mapv_inplace(|x| x.conjugate());
    // }

    fn khemv(op: &Self::OpRep, alpha: N,
             x: & Self::KetRep, y: &mut Self::KetRep, beta: N){
        ndarray::linalg::general_mat_vec_mul(alpha, op, x, beta, y );
    }

    fn kscal(a: N, ket: &mut Self::KetRep){
        ket.mapv_inplace(|q| q*a);
    }

    // fn qscal(a: N, op: &mut Self::OpRep) {
    //     op.mapv_inplace(|q| q*a);}

    fn kaxpy(a: N, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep){
        ket_y.zip_mut_with(ket_x, |y,&x| *y += a* x );
    }

}

// impl<N: Scalar > EigQRep<N> for DenseQRep<N>
//     where N: ComplexField<RealField=<N as cauchy::Scalar>::Real>,
//         <N as cauchy::Scalar>::Real : RealField
// {
//     fn eig(op: &Self::OpRep) -> (Vec<N::Real>, Self::OpRep) {
//         let (vals, vecs) = op.eigh(UPLO::Upper).unwrap();
//
//         (vals.into_raw_vec(), vecs)
//     }
// }

impl<N, S, D> LinearCombination<N, ArrayBase<S, D>> for LC<N>
where   N: Scalar,
        S: ndarray::DataMut<Elem=N>,
        D: ndarray::Dimension
{
    fn scale(v: &mut ArrayBase<S, D>, k: N) {
        v.map_inplace(|vi| *vi *= k);
       //*v *= k;
    }

    fn scalar_multiply_to(v: &ArrayBase<S, D>, k: N, target: &mut ArrayBase<S, D>) {
        target.zip_mut_with(v, |t, vi| *t = *vi * k);
    }

    fn add_scalar_mul(v: &mut ArrayBase<S, D>, k: N, other: &ArrayBase<S, D>) {
        v.zip_mut_with(other, move |y, x| *y = *y + (k * *x));
    }

    fn add_assign_ref(v: &mut ArrayBase<S, D>, other: &ArrayBase<S, D>) {
        *v += other;
    }

    fn delta(v: &mut ArrayBase<S, D>, y: &ArrayBase<S, D>) {
        *v -= y;
    }
}

impl<
    N: Scalar>
QObj<N>
for Array2<N>
{
    type Rep = DenseQRep<N>;
    type Dims = (usize, usize);

    fn qdim(&self) -> Self::Dims {
        let sh = self.shape();
        (sh[0], sh[1])
    }

    fn qtype(&self) -> QType {
        let sh = self.qdim();
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
        self.zip_mut_with(x, |yi, &xi| *yi += a*xi )
    }

    fn qscal(&mut self, a: N) {
        self.mapv_inplace(|xi| a*xi);
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        if b.is_zero(){
            self.zip_mut_with(x,|yi, &xi| *yi = a*xi);
        } else {
            self.zip_mut_with(x,  |yi, &xi| *yi = a*xi + b * *yi);
        }
    }
}
impl<
    N: Scalar >
QObj<N>
for Array1<N>
{
    type Rep = DenseQRep<N>;
    type Dims = usize;

    fn qdim(&self) -> Self::Dims {
        let sh = self.shape();
        sh[0]
    }
    fn qtype(&self) -> QType {
        QType::QKet
    }

    fn qaxpy(&mut self, a: N, x: &Self) {
        self.zip_mut_with(x, |yi, &xi| *yi += a*xi )
    }

    fn qscal(&mut self, a: N) {
        self.mapv_inplace(|xi| a*xi);
    }

    fn qaxby(&mut self, a: N, x: &Self, b: N) {
        if b.is_zero(){
            self.zip_mut_with(x,|yi, &xi| *yi = a*xi);
        } else {
            self.zip_mut_with(x,  |yi, &xi| *yi = a*xi + b * *yi);
        }
    }
}

impl<N: Scalar>
TensorProd<N, Op<N> >
for Op<N>
{
    type Result = Op<N>;

    fn tensor(a: Self, b:  Op<N>) -> Self::Result{
        kronecker(a.view(), b.view())
    }

    fn tensor_ref(a: &Self, b: & Op<N>)-> Self::Result{
        kronecker(a.view(), b.view())
    }
}

pub fn tensor_list<N: Scalar>(ops: &[ Op<N>]) -> Op<N>{
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

impl<N: Scalar>
TensorProd<N, Ket<N>>
for Ket<N>{
    type Result = Ket<N>;

    fn tensor(a: Self, b:  Ket<N>) -> Self::Result{
        TensorProd::tensor_ref(&a, &b)
        // kronecker(a.insert_axis(Axis(0)).view(), b.insert_axis(Axis(0)).view())
        //     .index_axis_move(Axis(0),0)
    }

    fn tensor_ref(a: &Self, b: & Ket<N>)-> Self::Result{
        kronecker(a.view().insert_axis(Axis(0)), b.view().insert_axis(Axis(0)))
            .index_axis_move(Axis(0),0)
    }
}

impl<N: Scalar>
TensorProd<N, Bra<N>>
for Bra<N>{
    type Result = Bra<N>;

    fn tensor(a: Self, b:  Bra<N>) -> Self::Result{
        Self{q: TensorProd::tensor(a.q, b.q)}
    }

    fn tensor_ref(a: &Self, b: & Bra<N>)-> Self::Result{
        Self{q: TensorProd::tensor_ref(&a.q, &b.q)}
    }
}

impl<N: Scalar> QOp<N> for Op<N>
{
    //type Rep = DenseQRep<N>;
}

impl<N: Scalar> QKet<N> for Ket<N>
{
    //type Rep = DenseQRep<N>;
}

impl<N: Scalar> QBra<N> for Bra<N>
{
    //type Rep = DenseQRep<N>;
}

#[cfg(test)]
mod tests{
    use crate::quantum::qdot;

    use super::{Bra, Ket, QBra, QKet};

    #[test]
    fn test_dense_qrep(){
        let a : Bra<f64> = Bra::from(Ket::zeros(3));
        let b : Ket<f64> = Ket::zeros(3);
        //b.qdot(a);
        let c = QBra::qdot(&a, &b);
        let d = qdot(&a, &b);
    }
}
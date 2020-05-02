pub use qrs_core::reps::dense::*;

// use super::quantum::*;
// use crate::util::eig_resolver::{EigResolver, EigJob, EigRangeData};
// use crate::util::array::{kronecker};
// use alga::general::{RealField, ComplexField};
// use num_complex::Complex;
// use ndarray::prelude::*;
//
// use std::marker::PhantomData;
// use crate::ComplexScalar;
//
//
// pub type Ket<N> = Array1<N>;
// pub type Bra<N> = ConjugatingWrapper<Array1<N>>;
// pub type Op<N> =  Array2<N>;
//
// pub struct DenseQRep<R: RealField, N>
//     where N: ComplexScalar<R>
// { _phantom: PhantomData<(R,N)> }
//
// impl<R: RealField, N: ComplexScalar<R> > QRep<R, N> for DenseQRep<R, N>
// //where Complex<R>: ComplexScalar<R>
// {
//     type KetRep = Ket<N>;
//     type BraRep = Bra<N>;
//     type OpRep = Op<N>;
//
//     fn qdim_op(op: &Self::OpRep) -> (usize, usize) {
//         let shape = op.shape();
//         return (shape[0], shape[1])
//     }
//
//     fn qdim_ket(ket: &Self::KetRep) -> usize {
//         return ket.len();
//     }
//
//     fn qdim_bra(bra: &Self::BraRep) -> usize {
//         return bra.q.len();
//     }
//
//
//     fn qbdot(bra: &Self::BraRep, ket: & Self::KetRep) -> N{
//         bra.q.dot(ket)
//     }
//
//     fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> N {
//         u.iter().zip(v.iter())
//             .map(|(&ui,&vi)| N::conjugate(ui) * vi)
//             .fold(N::zero(), |d, xi|d + xi) // sum not autoimpl
//     }
//
//     fn qswap(bra: &mut Self::BraRep, ket: & mut Self::KetRep){
//         for (q,r) in bra.iter_mut().zip(
//             ket.iter_mut()){
//             std::mem::swap(q, r);
//         }
//         bra.q.mapv_inplace(|x| x.conjugate());
//         ket.mapv_inplace(|x| x.conjugate());
//     }
//
//     fn khemv(op: &Self::OpRep, alpha: N,
//              x: & Self::KetRep, y: &mut Self::KetRep, beta: N){
//         ndarray::linalg::general_mat_vec_mul(alpha, op, x, beta, y );
//     }
//
//     fn kscal(a: N, ket: &mut Self::KetRep){
//         ket.mapv_inplace(|q| q*a);
//     }
//
//     fn qscal(a: N, op: &mut Self::OpRep) {
//         op.mapv_inplace(|q| q*a);}
//
//     fn kaxpy(a: N, ket_x: &Self::KetRep, ket_y: &mut Self::KetRep){
//         ket_y.zip_mut_with(ket_x, |y,&x| *y += a* x );
//     }
//
//     fn eig(op: & Self::OpRep) -> (Vec<R>, Self::OpRep){
//         let shape = Self::qdim_op(op);
//         assert_eq!(shape.0, shape.1);
//
//
//         let mut eiger:EigResolver<N> = EigResolver::new_eiger(
//             shape.0 as u32, EigJob::ValsVecs,
//             EigRangeData::<R>::all()
//         );
//         let m = eiger.borrow_matrix();
//         m.zip_mut_with(op,|x, &y| *x = y);
//         let (vals, vecs) = eiger.into_eigs();
//         let vals: Vec<R> = vals.into_raw_vec();
//
//         (vals, vecs)
//     }
// }
//
//
// impl<NR: RealField,
//     N: ComplexScalar<NR> >
// QObj<NR, N>
// for Array2<N>
// {
//     type Rep = DenseQRep<NR, N>;
//     type Dims = (usize, usize);
//
//     fn qdim(&self) -> Self::Dims {
//         let sh = self.shape();
//         (sh[0], sh[1])
//     }
//
//     fn qtype(&self) -> QType {
//         let sh = self.qdim();
//         match sh{
//             (1, 1) => QType::QScal,
//             (1, n) => QType::QKet,
//             (n, 1) => QType::QBra,
//             (n, m) =>{
//                 QType::QOp(QOpType::Ge)
//             }
//         }
//         //return QType::QObj;
//     }
//
//     fn qaxpy(&mut self, a: N, x: &Self) {
//         self.zip_mut_with(x, |yi, &xi| *yi += a*xi )
//     }
//
//     fn qscal(&mut self, a: N) {
//         self.mapv_inplace(|xi| a*xi);
//     }
//
//     fn qaxby(&mut self, a: N, x: &Self, b: N) {
//         if b.is_zero(){
//             self.zip_mut_with(x,|yi, &xi| *yi = a*xi);
//         } else {
//             self.zip_mut_with(x,  |yi, &xi| *yi = a*xi + b * *yi);
//         }
//     }
// }
// impl<NR: RealField,
//     N: ComplexScalar<NR> >
// QObj<NR, N>
// for Array1<N>
// {
//     type Rep = DenseQRep<NR, N>;
//     type Dims = usize;
//
//     fn qdim(&self) -> Self::Dims {
//         let sh = self.shape();
//         sh[0]
//     }
//     fn qtype(&self) -> QType {
//         QType::QKet
//     }
//
//     fn qaxpy(&mut self, a: N, x: &Self) {
//         self.zip_mut_with(x, |yi, &xi| *yi += a*xi )
//     }
//
//     fn qscal(&mut self, a: N) {
//         self.mapv_inplace(|xi| a*xi);
//     }
//
//     fn qaxby(&mut self, a: N, x: &Self, b: N) {
//         if b.is_zero(){
//             self.zip_mut_with(x,|yi, &xi| *yi = a*xi);
//         } else {
//             self.zip_mut_with(x,  |yi, &xi| *yi = a*xi + b * *yi);
//         }
//     }
// }
//
// impl<NR: RealField, N: ComplexScalar<NR>>
// TensorProd<NR, N, Op<N> >
// for Op<N>
// {
//     type Result = Op<N>;
//
//     fn tensor(a: Self, b:  Op<N>) -> Self::Result{
//         kronecker(a.view(), b.view())
//     }
//
//     fn tensor_ref(a: &Self, b: & Op<N>)-> Self::Result{
//         kronecker(a.view(), b.view())
//     }
// }
//
// pub fn tensor_list<R: RealField, N: ComplexScalar<R>>(ops: &[ Op<N>]) -> Op<N>{
//     if ops.len() == 0{
//         panic!("tensor_list Op slice cannot be empty");
//     }
//     let (first, rest) = ops.split_at(1);
//     let mut v = first[0].clone();
//     for u in rest.iter(){
//         v = TensorProd::tensor_ref(&v, &u)
//     }
//
//     v
// }
//
// impl<NR: RealField, N: ComplexScalar<NR>>
// TensorProd<NR, N, Ket<N>>
// for Ket<N>{
//     type Result = Ket<N>;
//
//     fn tensor(a: Self, b:  Ket<N>) -> Self::Result{
//         TensorProd::tensor_ref(&a, &b)
//         // kronecker(a.insert_axis(Axis(0)).view(), b.insert_axis(Axis(0)).view())
//         //     .index_axis_move(Axis(0),0)
//     }
//
//     fn tensor_ref(a: &Self, b: & Ket<N>)-> Self::Result{
//         kronecker(a.view().insert_axis(Axis(0)), b.view().insert_axis(Axis(0)))
//             .index_axis_move(Axis(0),0)
//     }
// }
//
// impl<N: RealField>
// TensorProd<Complex<N>, Bra<N>>
// for Bra<N>{
//    type Result = Bra<N>;
//
//    fn tensor(a: Self, b:  Bra<N>) -> Self::Result{
//        a.q.kronecker(&b.q)
//    }
//
//    fn tensor_ref(a: &Self, b: & Bra<N>)-> Self::Result{
//        a.q.kronecker(&b.q)
//    }
// }
//
// impl<R: RealField, N: ComplexScalar<R>> QOp<R, N> for Op<N>
// {
//     //type Rep = DenseQRep<N>;
// }
//
// impl<R: RealField, N: ComplexScalar<R>> QKet<R, N> for Ket<N>
// {
//     //type Rep = DenseQRep<N>;
// }
//
// impl<R: RealField, N: ComplexScalar<R>> QBra<R, N> for Bra<N>
// {
//     //type Rep = DenseQRep<N>;
// }
//
// #[cfg(test)]
// mod tests{
//     use super::{Bra, Ket, QKet, QBra};
//     use crate::base::quantum::qdot;
//     #[test]
//     fn test_dense_qrep(){
//         let a : Bra<f64> = Bra::from(Ket::zeros(3));
//         let b : Ket<f64> = Ket::zeros(3);
//         //b.qdot(a);
//         let c = QBra::qdot(&a, &b);
//         let d = qdot(&a, &b);
//     }
// }
//extern crate qrs;
//extern crate alga;
//extern crate nalgebra;
//use alga::general::ClosedAdd;
//use nalgebra::base::{Matrix, DMatrix, Scalar, Dim};
////use qrs::algebra::expressions::{Expression, ExpressionEval, ExpressionAtom, BinaryOpExpression};
//
//trait AssignIterator : Iterator{
//
//}
//
//
//fn scal_mul_assign<N: Scalar, R: Dim, C: Dim, S>( to_mat: &mut Matrix<N, R, C, S>){
//    to_mat.iter_mut()
//        .map(|&mut x| x += 1)
//
//}
//
//struct MatrixExpression<'a, N: Scalar>{
//    _mat: &'a DMatrix<N>
//}
//
//impl<'a, N: Scalar> Expression<&'a DMatrix<N>> for MatrixExpression<'a, N>{
//
//}
//
//impl<'a, N: Scalar> ExpressionAtom<&'a DMatrix<N>> for MatrixExpression<'a, N>{
//    fn into_object(self) -> &'a DMatrix<N>{
//        self._mat
//    }
//}
//
//impl<'a, N: Scalar> ExpressionEval<&'a DMatrix<N>, usize, &'a N> for MatrixExpression<'a, N>{
//    unsafe fn eval(&self, idx: usize) -> &'a N{
//        self._mat.get_unchecked(idx)
//    }
//}
//
//
//struct MatrixSumExpression<'a, N: Scalar, LHS, RHS>
//    where LHS: ExpressionEval<&'a DMatrix<N>, usize, &'a N>,
//          RHS: ExpressionEval<&'a DMatrix<N>, usize, &'a N>{
//    _lhs: LHS,
//    _rhs: RHS
//}
//
//impl<'a, N: Scalar, LHS, RHS> Expression<&'a DMatrix<N>>
//for MatrixSumExpression<'a, N, LHS, RHS>
//where LHS: Expression<&'a DMatrix<N>>, RHS: Expression<&'a DMatrix<N>>{ }
//
//impl<'a, N: Scalar, LHS, RHS> ExpressionEval<&'a DMatrix<N>, usize, N>
//for MatrixSumExpression<'a, N, LHS, RHS>
//where LHS: ExpressionEval<&'a DMatrix<N>, usize, &'a N>,
//      RHS: ExpressionEval<&'a DMatrix<N>, usize, &'a N>{
//    unsafe fn eval(&self, idx: usize) -> N{
//        self._rhs.eval(idx) + self._lhs.eval(idx)
//    }
//}
//
//impl<'a, N: Scalar+ClosedAdd, LHS, RHS> BinaryOpExpression<&'a DMatrix<N>, usize, N>
//for MatrixSumExpression<'a, N, LHS, RHS>{
//    type Rhs = RHS;
//    type Lhs = RHS;
//    fn binary_op(&self, t1: N, t2: N) -> N{
//        t1 + t2
//    }
//    fn split_src(&self, idx: usize) -> (usize, usize){
//        (idx, idx)
//    }
//    fn rhs(&self) -> &RHS{
//        &self._rhs
//    }
//    fn lhs(&self) -> &LHS{
//        &self._lhs
//    }
//
//}
//
//
//fn main(){
//
//}
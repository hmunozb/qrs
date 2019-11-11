use alga::general::{ Module, ClosedMul, ClosedAdd};
use alga::general::{ RingCommutative};
//use num_traits::{Zero, One};
use std::ops::{Add, Mul, AddAssign};
use std::boxed::Box;
use std::marker::PhantomData;
use std::mem;
//use crate::algebra::algebra::{Algebra};
use std::borrow::BorrowMut;


trait FreeExpression<S>{
    fn into_object(self) -> S;
}

trait FreeExpressionBinaryOp<S> : FreeExpression<S>{
    type Rhs : FreeExpression<S>;
    type Lhs : FreeExpression<S>;
    fn binary_op(self, s1: S, s2: S) -> S;
    fn rhs(&self) -> &Self::Rhs;
    fn lhs(&self) -> &Self::Lhs;
}

trait BinaryOpFunctor{

}

//trait FreeExpressionPropagator<S> : FreeExpression<S>{
//    type Propaganda;
//    type Response;
//
//    fn propagate(&self, p: Propaganda) -> Response;
//}
trait FreeExpressionAtom<S>: FreeExpression<S>{
    fn atom(&self) -> &S;
}
trait FreeExpressionSum<S, A1: FreeExpression<S>, A2: FreeExpression<S>> : FreeExpression<S>{
    fn expression_add(a1: A1, a2: A2) -> Self;
}
trait FreeExpressionProduct<S, A1: FreeExpression<S>, A2: FreeExpression<S>> : FreeExpression<S>{
    fn expression_mul(a1: A1, a2: A2) -> Self;
}

pub enum DynFreeAlgebra<S:RingCommutative+Module , T: RingCommutative = <S as Module>::Ring>{
    Zero,
    Unity,
    Element(S),
    Sum(Box<DynFreeAlgebra<S, T>>, Box<DynFreeAlgebra<S, T>>),
    Prod(Box<DynFreeAlgebra<S, T>>, Box<DynFreeAlgebra<S, T>>),
    Scale(Box<DynFreeAlgebra<S, T>>, T)
}

/// Allocate an empty instance of the enum and swap its contents with self,
fn push_dfa_out<S:RingCommutative+Module, T: RingCommutative>(
         a: &mut DynFreeAlgebra<S, T>) -> Box<DynFreeAlgebra<S, T>>{
    let mut lhs_box = Box::from(DynFreeAlgebra::Zero);
    mem::swap(a, lhs_box.borrow_mut());
    lhs_box
}

impl<S:RingCommutative+Module+ClosedMul<T>, T: RingCommutative> FreeExpression<S> for DynFreeAlgebra<S, T>{
    fn into_object(self) -> S{
        match self{
            DynFreeAlgebra::Zero => S::zero(),
            DynFreeAlgebra::Unity => S::one(),
            DynFreeAlgebra::Element(s) => s,
            DynFreeAlgebra::Sum(a1, a2) =>
                a1.into_object() + a2.into_object(),
            DynFreeAlgebra::Prod(a1, a2) =>
                a1.into_object() * a2.into_object(),
            DynFreeAlgebra::Scale(a, t) =>
                a.into_object() * t
        }
    }
}



impl<S:RingCommutative+Module, T:RingCommutative> Add
for DynFreeAlgebra<S, T>{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output{
        DynFreeAlgebra::Sum(Box::from(self), Box::from(rhs))
    }
}

impl<S:RingCommutative+Module, T:RingCommutative> AddAssign
for DynFreeAlgebra<S, T>{
    fn add_assign(&mut self, rhs: Self) {
//        let self_p = self as * mut Self;
//        let lhs_box = Box::from(DynFreeAlgebra::Zero);
//        //Allocate an empty instance of the enum, swap contents with self, and turn self into
//        //a sum
//        unsafe{
//            let lhs_p = Box::into_raw(lhs_box);
//            ptr::swap(self_p, lhs_p);
//            *self = DynFreeAlgebra::Sum(Box::from_raw(lhs_p), Box::from(rhs));
//        }
        let lhs_box = push_dfa_out(self);
        *self = DynFreeAlgebra::Sum(lhs_box, Box::from(rhs))
    }
}


struct StaticFreeExpressionAtom<S>{
    atom: S
}
struct StaticFreeExpressionSum<S: Add<S>, A1: FreeExpression<S>, A2: FreeExpression<S>>{
    a1: A1,
    a2: A2,
    _phantom: PhantomData<S>
}
struct StaticFreeExpressionProduct<S: Mul<S>, A1: FreeExpression<S>, A2: FreeExpression<S>>{
    a1: A1,
    a2: A2,
    _phantom: PhantomData<S>
}
struct StaticFreeExpressionScale<T, S: Mul<T>, A: FreeExpression<S>>{
    t: T,
    a: A,
    _phantom: PhantomData<S>
}
impl<S> FreeExpression<S> for StaticFreeExpressionAtom<S>{
    fn into_object(self) -> S{
        self.atom
    }
}
impl<S: ClosedAdd, A1: FreeExpression<S>, A2: FreeExpression<S>> FreeExpression<S>
for StaticFreeExpressionSum<S, A1, A2>{
    fn into_object(self) ->S {
        self.a1.into_object() + self.a2.into_object()
    }
}
impl<S: ClosedMul, A1: FreeExpression<S>, A2: FreeExpression<S>> FreeExpression<S>
for StaticFreeExpressionProduct<S, A1, A2>{
    fn into_object(self) -> S{
        self.a1.into_object() * self.a2.into_object()
    }
}
impl<T, S: ClosedMul<T>, A: FreeExpression<S>> FreeExpression<S>
for StaticFreeExpressionScale<T, S, A>{
    fn into_object(self) -> S{
        self.a.into_object() * self.t
    }
}

// Free Expression Add Implementations
impl<S: ClosedAdd<S>, A1: FreeExpression<S>, A2: FreeExpression<S>> FreeExpressionSum<S, A1, A2>
for StaticFreeExpressionSum<S, A1, A2>{
    fn expression_add(a1: A1, a2: A2) -> Self{
        StaticFreeExpressionSum{a1, a2, _phantom: PhantomData}
    }
}

impl<S: ClosedAdd<S>> Add<StaticFreeExpressionAtom<S>> for StaticFreeExpressionAtom<S>{
    type Output = StaticFreeExpressionSum<S, StaticFreeExpressionAtom<S>, StaticFreeExpressionAtom<S>>;

    fn add(self, rhs: Self) -> Self::Output{
        StaticFreeExpressionSum{a1: self, a2: rhs, _phantom: PhantomData}
    }
}
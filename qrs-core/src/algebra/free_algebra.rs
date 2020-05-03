
use std::borrow::BorrowMut;
use std::boxed::Box;
use std::marker::PhantomData;
use std::mem;

use std::ops::{Add, AddAssign, DerefMut, Mul, MulAssign};

use num_traits::{One, Zero};

use crate::{ClosedAdd, ClosedMul};

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

pub enum DynVectorSpaceSubElement<S: Sized, T: Sized>{
    Zero,
    Element(S),
    ScaledElement(S, T)
}
impl<S: Sized, T: Sized> DynVectorSpaceSubElement<S, T>{

}
//
//impl<S: Sized, T: Sized> Into<DynVectorSpace<S, T>> for DynVectorSpaceSubElement<S, T>{
//    fn into(self) -> DynVectorSpace<S, T> {
//        let mut me = self;
//        match me{
//            DynVectorSpaceSubElement::Zero => DynVectorSpace::Zero,
//            DynVectorSpaceSubElement::Element(s) => DynVectorSpace::Element(s),
//            DynVectorSpaceSubElement::Boxed(mut b) => {
//                let mut d = DynVectorSpace::Zero;
//                mem::swap(&mut d, &mut b);
//                d
//            }
//        }
//    }
//}


pub trait DynVectorSpaceZero{
    fn zero() -> Self;
    fn is_zero(&self) -> bool
        where Self: PartialEq;
}

impl<T> DynVectorSpaceZero for T where T: Zero{
    fn zero() -> Self {
        Zero::zero()
    }

    fn is_zero(&self) -> bool
    where Self: PartialEq{
        Zero::is_zero(self)
    }
}

pub trait DynAlgebraOne{
    fn one() -> Self;
    fn is_one(&self) -> bool
        where Self: PartialEq;
}

impl<T> DynAlgebraOne for T where T: One{
    fn one() -> Self {
        One::one()
    }

    fn is_one(&self) -> bool
    where Self: PartialEq{
        One::is_one(self)
    }
}

pub struct DynLinearCombination<S: Sized, T: Sized>{
    v: Vec<DynVectorSpaceSubElement<S, T>>
}
impl<S: Sized, T: Sized> DynLinearCombination<S, T>{
    pub fn linear_map<F, S2>(self, f: &F)
        where F: Fn(S) -> S2, S2: Sized{

    }
}

/// Generic, dynamically allocated vector space structure
#[derive(Clone, Debug)]
pub enum DynVectorSpace<S: Clone, T: Copy>{
    Zero,
    Element(S),
    Add(Box<DynVectorSpace<S, T>>, Box<DynVectorSpace<S, T>>),
    Sum(Vec<DynVectorSpace<S, T>>),
    Scale(Box<DynVectorSpace<S, T>>, T)
}

impl<S, T>  DynVectorSpace<S, T>
    where S: Clone, T: Copy
{
    pub fn evaluate<F, A>(self, f: &F, zero: &A) -> A
        where F: Fn(S) -> A,
              A: Add<Output=A>+Mul<T, Output=A> + Clone
    {
        match self{
            DynVectorSpace::Zero => zero.clone(),
            DynVectorSpace::Element(s) => f(s),
            DynVectorSpace::Add(a1, a2) =>
                a1.evaluate(f, zero) + a2.evaluate(f, zero),
            DynVectorSpace::Sum(v) =>
                v.into_iter().map(|a| a.evaluate(f, zero))
                    .fold(zero.clone(), |a, b| a + b),
            DynVectorSpace::Scale(a, t) =>
                a.evaluate(f, zero) * t
        }
    }

    pub fn evaluate_ref<F, G, A, T2>(&self, f: &F, g:& G, zero: &A) -> A
        where F: Fn(&S) -> A,
              G: Fn(&T) -> T2,
              A: Add<Output=A>+Mul<T2, Output=A> + Clone,
              T2: Copy
    {
        match self{
            DynVectorSpace::Zero => zero.clone(),
            DynVectorSpace::Element(s) => f(s),
            DynVectorSpace::Add(a1, a2) =>
                a1.evaluate_ref(f, g, zero) + a2.evaluate_ref(f, g,zero),
            DynVectorSpace::Sum(v) =>
                v.into_iter().map(|a| a.evaluate_ref(f, g, zero))
                    .fold(zero.clone(), |a, b| a + b),
            DynVectorSpace::Scale(a, t) =>
                a.evaluate_ref(f,g,zero) * g(t)
        }
    }

    pub fn linear_map_ref<F, G, S2>(&self, f: &F, g: &G) -> DynVectorSpace<S2, T>
        where F: Fn(&S) -> S2,
              G: Fn(&T) -> T,
              S2: Clone
    {
        match self{
            DynVectorSpace::Zero => DynVectorSpace::Zero,
            DynVectorSpace::Element(s) => DynVectorSpace::Element(f(s)),
            DynVectorSpace::Add(a1, a2) =>
                DynVectorSpace::Add(Box::from(a1.linear_map_ref(f, g)), Box::from(a2.linear_map_ref(f, g))),
            DynVectorSpace::Sum(v) =>
                DynVectorSpace::Sum(v.iter().map(|a| a.linear_map_ref(f, g)).collect())
                ,
            DynVectorSpace::Scale(a, t) =>
               DynVectorSpace::Scale(Box::from(a.linear_map_ref(f, g)), g(t))
        }
    }

    pub fn linear_map<F, G, S2>(self, f: &F, g: &G) -> DynVectorSpace<S2, T>
        where F: Fn(S) -> S2,
              G: Fn(T) -> T,
              S2: Clone
    {
        match self{
            DynVectorSpace::Zero => DynVectorSpace::Zero,
            DynVectorSpace::Element(s) => DynVectorSpace::Element(f(s)),
            DynVectorSpace::Add(a1, a2) =>
                DynVectorSpace::Add(Box::from(a1.linear_map(f, g)), Box::from(a2.linear_map(f, g))),
            DynVectorSpace::Sum(v) =>
                DynVectorSpace::Sum(v.into_iter().map(|a| a.linear_map(f, g)).collect())
            ,
            DynVectorSpace::Scale(a, t) =>
                DynVectorSpace::Scale(Box::from(a.linear_map(f, g)), g(t))
        }
    }

    pub fn linear_apply<F>(&mut self, f: &F)
        where F: Fn(&mut S),
    {
        match self{
            DynVectorSpace::Element(s) => f(s),
            DynVectorSpace::Add(a1, a2) =>
                { a1.linear_apply(f); a2.linear_apply(f);}
            DynVectorSpace::Sum(v) =>
                for a in v.iter_mut(){
                    a.linear_apply(f);
                },
            DynVectorSpace::Scale(a, _t) =>
                { a.linear_apply(f);}
            _ => ()
        };
    }
//    fn leftmost_split(self){
//        if let DynVectorSpace::Add(a, b) = self{
//
//        }
//    }
//    /// Left-reassociate for Add(self, rhs)
//    fn reassociate_with(self, rhs: Self){
//        if let DynVectorSpace::Add(a, b) = self{
//            if let DynVectorSpace::Add(c, d) = rhs{
//
//            }
//        }
//
//    }
//
//    pub fn reassociate(self) -> Self{
//        match self{
//            DynVectorSpace::Zero => DynVectorSpace::Zero,
//            DynVectorSpace::Element(s) => DynVectorSpace::Element(s),
//            DynVectorSpace::Add(a1, a2) =>{
//
//            }
//
//        }
//    }
}
impl<S, T>  DynVectorSpace<S, T>
    where S: Clone+DynVectorSpaceZero+PartialEq,T: Copy + Mul + Zero
{
    pub fn simplify(self) -> Self
    {
        let me = self;
        match me{
            DynVectorSpace::Zero => DynVectorSpace::Zero,
            DynVectorSpace::Element(s) => {
                if s.is_zero(){
                    DynVectorSpace::Zero
                } else {
                    DynVectorSpace::Element(s)
                }
            }
            DynVectorSpace::Add(a1, a2) =>{
                if let DynVectorSpace::Zero = *a1{
                    a2.simplify()
                } else if let DynVectorSpace::Zero = *a2{
                    a1.simplify()
                } else {
                    let a1 = a1.simplify();
                    let a2 = a2.simplify();
                    DynVectorSpace::Add(Box::from(a1), Box::from(a2))
                }
            }
            DynVectorSpace::Sum(v) =>{
                DynVectorSpace::Sum(v.into_iter().map(|a|a.simplify()).collect())
            }
            DynVectorSpace::Scale(mut a, t) =>{
                if t.is_zero(){
                    return DynVectorSpace::Zero
                };

                let mut d = DynVectorSpace::Zero;
                mem::swap(a.deref_mut(), &mut d);
                match d { //Look-ahead at the object for possible rescaling simplifications
                    DynVectorSpace::Zero => DynVectorSpace::Zero,
                    DynVectorSpace::Element(s) =>  DynVectorSpace::Element(s),
                    DynVectorSpace::Scale(mut b, t2) =>{
                        if t2.is_zero(){
                            DynVectorSpace::Zero
                        } else {
                            let mut b2 = DynVectorSpace::Zero;
                            mem::swap(&mut b2, b.deref_mut());
                            let b2 = b2.simplify();
                            if let &DynVectorSpace::Zero = &b2{
                                return DynVectorSpace::Zero
                            };
                            let b = Box::new(b2);
                            DynVectorSpace::Scale(b, t2)
                        }
                    }
                    DynVectorSpace::Add(a, b) => {
                        let d = DynVectorSpace::Add(a, b);
                        DynVectorSpace::Scale(Box::from(d.simplify()), t)
                    }
                    DynVectorSpace::Sum(v) =>{
                        let d = DynVectorSpace::Sum(v.into_iter().map(|a|a.simplify()).collect());
                        DynVectorSpace::Scale(Box::from(d), t)
                    }
                }
//                if let DynVectorSpace::Scale(b, t2) = d {
//                    if t2.is_zero(){
//                        DynVectorSpace::Zero
//                    } else {
//
//                    }
//                    DynVectorSpace::Scale(b, t * t2)
//                } else {
//                    let b = Box::new(d.simplify());
//                    DynVectorSpace::Scale(b, t)
//                }

            }
        }
    }
}

impl<S: Clone, T: Copy> Zero for DynVectorSpace<S, T>{
    fn zero() -> Self {
        DynVectorSpace::Zero
    }

    fn is_zero(&self) -> bool {
        if let &DynVectorSpace::Zero = &self{
            true
        } else {
            false
        }
    }
}

impl<S: Clone, T: Copy> Default for DynVectorSpace<S, T>{
    fn default() -> Self {
        Zero::zero()
    }
}

impl<S: Clone, T: Copy> Add
for DynVectorSpace<S, T>{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output{
        DynVectorSpace::Add(Box::from(self), Box::from(rhs))
    }
}

impl<S: Clone, T: Copy> Add<&DynVectorSpace<S, T>>
for DynVectorSpace<S, T>{
    type Output = Self;
    fn add(self, rhs: &DynVectorSpace<S, T>) -> Self::Output{
        DynVectorSpace::Add(Box::from(self), Box::from((*rhs).clone()))
    }
}

impl<S: Clone, T: Copy> AddAssign
for DynVectorSpace<S, T>{
    fn add_assign(&mut self, rhs: Self) {
        let mut lhs_box = Box::from(DynVectorSpace::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynVectorSpace::Add(lhs_box, Box::from(rhs))
    }
}


impl<S: Clone, T: Copy> AddAssign<&DynVectorSpace<S, T>>
for DynVectorSpace<S, T>{
    fn add_assign(&mut self, rhs: &DynVectorSpace<S, T>) {
        let mut lhs_box = Box::from(DynVectorSpace::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynVectorSpace::Add(lhs_box, Box::from((*rhs).clone()))
    }
}

impl<S: Clone, T: Copy> Mul<T>
for DynVectorSpace<S, T>{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output{
        DynVectorSpace::Scale(Box::from(self), rhs)
    }
}

impl<S: Clone, T: Copy> MulAssign<T>
for DynVectorSpace<S, T>{
    fn mul_assign(&mut self, rhs: T) {
        let mut lhs_box = Box::from(DynVectorSpace::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynVectorSpace::Scale(lhs_box, rhs);
    }
}

pub enum DynFreeAlgebraMonomial<S: Sized>{
    Unity,
    Prod(Vec<S>)
}

/// Symbolic Enum for dynamically representing an algebra generated by a module S
#[derive(Clone, Debug)]
pub enum DynFreeAlgebra<S: Clone, T: Copy>{
    Zero,
    Unity,
    Element(S),
    Add(Box<DynFreeAlgebra<S, T>>, Box<DynFreeAlgebra<S, T>>),
    Prod(Box<DynFreeAlgebra<S, T>>, Box<DynFreeAlgebra<S, T>>),
    Scale(Box<DynFreeAlgebra<S, T>>, T)
}

/// Allocate an empty instance of the enum and swap its contents with self,
fn push_dfa_out<S: Clone, T: Copy>(
         a: &mut DynFreeAlgebra<S, T>) -> Box<DynFreeAlgebra<S, T>>{
    let mut lhs_box = Box::from(DynFreeAlgebra::Zero);
    mem::swap(a, lhs_box.borrow_mut());
    lhs_box
}

impl<S: Clone, T: Copy> DynFreeAlgebra<S, T>
{
    pub fn evaluate<F, A>(self, f: &F, zero: &A, one:&A) -> A
    where F: Fn(S) -> A,
          A: Add<Output=A>+Mul<Output=A>+Mul<T, Output=A> + Clone{
        match self{
            DynFreeAlgebra::Zero => zero.clone(),
            DynFreeAlgebra::Unity => one.clone(),
            DynFreeAlgebra::Element(s) => f(s),
            DynFreeAlgebra::Add(a1, a2) =>
                a1.evaluate(f, zero, one) + a2.evaluate(f, zero, one),
            DynFreeAlgebra::Prod(a1, a2) =>
                a1.evaluate(f,zero,one) * a2.evaluate(f, zero, one),
            DynFreeAlgebra::Scale(a, t) =>
                a.evaluate(f, zero, one) * t
        }
    }

    pub fn evaluate_ref<F, G, A, T2>(&self, f: &F, g: &G,
                                     zero: &A, one:&A) -> A
        where F: Fn(&S) -> A,
              G: Fn(&T) -> T2,
              A: Add<Output=A> + Mul<Output=A> + Mul<T2, Output=A> + Clone{
        match self{
            DynFreeAlgebra::Zero => zero.clone(),
            DynFreeAlgebra::Unity => one.clone(),
            DynFreeAlgebra::Element(s) => f(s),
            DynFreeAlgebra::Add(a1, a2) =>
                a1.evaluate_ref(f, g, zero, one) + a2.evaluate_ref(f, g, zero, one),
            DynFreeAlgebra::Prod(a1, a2) =>
                a1.evaluate_ref(f, g, zero,one) * a2.evaluate_ref(f, g, zero, one),
            DynFreeAlgebra::Scale(a, t) =>
                a.evaluate_ref(f, g, zero, one) * g(t)
        }
    }
}

impl<S: Clone, T: Copy> Add
for DynFreeAlgebra<S, T>{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output{
        DynFreeAlgebra::Add(Box::from(self), Box::from(rhs))
    }
}

impl<S: Clone, T: Copy> Add<&DynFreeAlgebra<S, T>>
for DynFreeAlgebra<S, T>{
    type Output = Self;
    fn add(self, rhs: &DynFreeAlgebra<S, T>) -> Self::Output{
        DynFreeAlgebra::Add(Box::from(self), Box::from((*rhs).clone()))
    }
}

impl<S: Clone, T: Copy> AddAssign
for DynFreeAlgebra<S, T>{
    fn add_assign(&mut self, rhs: Self) {
        let mut lhs_box = Box::from(DynFreeAlgebra::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynFreeAlgebra::Add(lhs_box, Box::from(rhs))
    }
}


impl<S: Clone, T: Copy> AddAssign<&DynFreeAlgebra<S, T>>
for DynFreeAlgebra<S, T>{
    fn add_assign(&mut self, rhs: &DynFreeAlgebra<S, T>) {
        let mut lhs_box = Box::from(DynFreeAlgebra::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynFreeAlgebra::Add(lhs_box, Box::from((*rhs).clone()))
    }
}

impl<S: Clone, T: Copy> Mul<T>
for DynFreeAlgebra<S, T>{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output{
        DynFreeAlgebra::Scale(Box::from(self), rhs)
    }
}

impl<S: Clone, T: Copy> MulAssign<T>
for DynFreeAlgebra<S, T>{
    fn mul_assign(&mut self, rhs: T) {
        let mut lhs_box = Box::from(DynFreeAlgebra::Zero);
        mem::swap(self, &mut lhs_box);
        *self = DynFreeAlgebra::Scale(lhs_box, rhs);
    }
}


//
//impl<S:Ring+Module+ClosedMul<T>, T: RingCommutative> FreeExpression<S> for DynFreeAlgebra<S, T>{
//    fn into_object(self) -> S{
//        match self{
//            DynFreeAlgebra::Zero => S::zero(),
//            DynFreeAlgebra::Unity => S::one(),
//            DynFreeAlgebra::Element(s) => s,
//            DynFreeAlgebra::Sum(a1, a2) =>
//                a1.into_object() + a2.into_object(),
//            DynFreeAlgebra::Prod(a1, a2) =>
//                a1.into_object() * a2.into_object(),
//            DynFreeAlgebra::Scale(a, t) =>
//                a.into_object() * t
//        }
//    }
//}



// impl<S: Clone, T: Copy> Add
// for DynFreeAlgebra<S, T>{
//     type Output = Self;
//     fn add(self, rhs: Self) -> Self::Output{
//         DynFreeAlgebra::Add(Box::from(self), Box::from(rhs))
//     }
// }

// impl<S: Clone, T: Copy> AddAssign
// for DynFreeAlgebra<S, T>{
//     fn add_assign(&mut self, rhs: Self) {
//         let lhs_box = push_dfa_out(self);
//         *self = DynFreeAlgebra::Add(lhs_box, Box::from(rhs))
//     }
// }


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
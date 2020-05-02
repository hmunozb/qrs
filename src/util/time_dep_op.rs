use crate::base::quantum::{QObj, QRep};
use crate::{ComplexScalar};
use qrs_core::{ComplexField,RealField, RealScalar};
use std::borrow::Cow;
use nalgebra::DMatrix;
use std::marker::PhantomData;


pub trait TimeDependentOperatorTermObj{
    type Operator;
    type Time;
    fn add_to(&self, out: &mut Self::Operator, t: Self::Time);
    fn eval_to(&self, out: &mut Self::Operator, t: Self::Time);
}

#[derive(Clone)]
pub struct TimeDependentOperatorTerm<
    'a, R: RealField, N: ComplexScalar,
    Q: QRep<N>, T: QObj<N, Rep=Q>>
{
    pub op: Cow<'a, T>,
    pub f: Box<&'a dyn Fn(R)->N>,
    _phantom: PhantomData<Q>
}

#[derive(Clone)]
pub struct TimeDependentOperator<'a, R: RealField, N: ComplexScalar,
    Q: QRep<N>, T: QObj<N, Rep=Q>>{
    pub terms: Vec<TimeDependentOperatorTerm<'a, R, N, Q, T>>,
}

//pub struct TimeDepOperatorTerm<N, NFn>{
//
//}

/// A Time Dependent Term consists of a matrix and a callable function f: R-> C
/// It is implemented with a Clone-on-write pointer to allow creating new time-dependent
/// terms by applying a linear function onto an existing time dependent matrix. The resulting
/// time-dependent terms own their matrix rather than a reference to a matrix
#[derive(Clone)]
pub struct TimeDepMatrixTerm<'a, N>
where N: ComplexField{
    pub mat: Cow<'a, DMatrix<N>>,
    pub f: Box<&'a dyn Fn(N::RealField)->N>
}

#[derive(Clone)]
pub struct TimeDepMatrix<'a, N: ComplexField> {
    pub terms: Vec<TimeDepMatrixTerm<'a, N>>
}

impl<'a, N: ComplexField> TimeDepMatrix<'a, N>{
    pub fn new(terms: Vec<TimeDepMatrixTerm<'a, N>>) -> Self{
        Self{terms}
    }
}

impl<'a, R: RealField, N: ComplexScalar, Q: QRep<N>, T: QObj<N, Rep=Q>>
TimeDependentOperatorTerm<'a, R, N, Q, T>
    where N: ComplexField{
    pub fn new(q: &'a T, f: &'a dyn Fn(R) -> N) -> Self{
        Self{op: Cow::Borrowed(q), f: Box::new(f), _phantom: Default::default() }
    }
    pub fn new_with_owned(q: T, f: &'a dyn Fn(R) -> N) -> Self{
        Self{op: Cow::Owned(q), f: Box::new(f), _phantom: Default::default() }
    }

    pub fn shape(&self) -> T::Dims{
        self.op.qdim()
        //return self.mat.shape()
    }

    pub fn add_to(& self, to_op: &mut T, t: R) {
        let ft = (self.f)(t);
        to_op.qaxpy(ft, &self.op);
    }

    pub fn eval_to(& self, to_op: &mut T, t: R){
        let ft = (self.f)(t);
        to_op.qaxby(ft, &self.op, N::zero());
    }
}

impl <'a, N> TimeDepMatrixTerm<'a, N>
    where N: ComplexField{
    pub fn new(mat: &'a DMatrix<N>, f: &'a dyn Fn(N::RealField) -> N) -> Self{
        Self{mat: Cow::Borrowed(mat), f: Box::new(f)}
    }
    pub fn new_with_owned(mat: DMatrix<N>, f: &'a dyn Fn(N::RealField) -> N) -> Self{
        Self{mat: Cow::Owned(mat), f: Box::new(f)}
    }

    pub fn shape(&self) -> (usize, usize){
        return self.mat.shape()
    }

    pub fn add_to(& self, to_mat: &mut DMatrix<N>, t: N::RealField) {
        let ft = (self.f)(t);
        to_mat.zip_apply(&self.mat, |x, y| x + ft * y);
//        // This operation can't be done directly without additional allocations
//        // *to_mat +=  (self.mat) * ft;
//        // so we use slice iterators directly
//        for (rhs, lhs) in self.mat.iter().zip(to_mat.iter_mut()){
//            *lhs += ft * (*rhs);
//        }
    }
    pub fn eval_to(& self, to_mat: &mut DMatrix<N>, t: N::RealField){
        let ft = (self.f)(t);
        to_mat.zip_apply(&self.mat, |_x, y| ft * y);
//        for (rhs, lhs) in self.mat.iter().zip(to_mat.iter_mut()){
//            *lhs = ft * (*rhs);
//        }
    }
}

impl<'a, N> TimeDepMatrix<'a, N>
    where N: ComplexField
{
    pub fn eval(&self, t: N::RealField) -> DMatrix<N>{
        let (r,c) = self.terms.first().unwrap().mat.shape();
        let mut m = DMatrix::zeros(r, c);
        self.eval_to(&mut m, t);

        m
    }

    pub fn shape(&self) -> (usize, usize){
        self.terms[0].shape()
    }

    pub fn eval_to(&self, to_mat: &mut DMatrix<N>, t: N::RealField){
        let (first, rest) = self.terms.split_at(1);
        first.get(0).unwrap().eval_to(to_mat, t);
        for term in rest.iter(){
            term.add_to(to_mat, t);
        }
    }

    pub fn map<Fun>(&self, f: Fun) -> Self
    where Fun: Fn(&DMatrix<N>) -> DMatrix<N>
    {
        let mut m = Self{terms: Vec::new()};
        for term in self.terms.iter(){
            m.terms.push(TimeDepMatrixTerm{mat: Cow::Owned(f(&&term.mat)), f: term.f.clone()})
        }

        m
    }
}

impl<'a, R: RealScalar, N: ComplexScalar<R=R>, Q: QRep<N>, T: QObj<N, Rep=Q>>
TimeDependentOperator<'a, R, N, Q, T>
{
    pub fn eval(&self, t: N::R) -> T{
        let mut op : T  = (*self.terms.first().unwrap().op).clone();
        self.eval_to(&mut op, t);
        op
    }

    pub fn shape(&self) -> T::Dims{
        self.terms[0].shape()
    }

    pub fn eval_to(&self, to_op: &mut T, t: R){
        let (first, rest) = self.terms.split_at(1);
        first.get(0).unwrap().eval_to(to_op, t);
        for term in rest.iter(){
            term.add_to(to_op, t);
        }
    }

    pub fn map<Fun>(&self, mut f: Fun) -> Self
        where Fun: FnMut(&T) -> T
    {
        let mut m = Self{terms: Vec::new()};
        for term in self.terms.iter(){
            m.terms.push(TimeDependentOperatorTerm{
                op: Cow::Owned(f(&&term.op)),
                f: term.f.clone(),
                _phantom: Default::default() })
        }

        m
    }
}

mod tests{
    #[test]
    fn test_time_dep_op(){
        use nalgebra::DMatrix;
        use crate::util::time_dep_op::{TimeDepMatrixTerm, TimeDepMatrix};
        use alga::general::ComplexField;
        use num_complex::Complex64 as c64;
        //use crate::base::DenseQRep;
        use crate::base::quantum::{TensorProd};

        let _1c = c64::new(1.0, 0.0);
        let _0c = c64::new(0.0, 0.0);
        let _i = c64::i();
        let _sx = [ _0c, _1c,
                    _1c, _0c];
        let _sz = [ _1c, _0c,
                    _0c, -_1c];

        let sx = DMatrix::from_row_slice(2, 2, &_sx);
        let sz = DMatrix::from_row_slice(2, 2, &_sz);
        let sx2: DMatrix<c64> = TensorProd::tensor_ref(&sx, &sx);
        let sz2: DMatrix<c64> = TensorProd::tensor_ref(&sz, &sz);
        let fx = |s:f64| c64::from_real(1.0 - s);
        let fz =|s: f64| c64::from_real(s);
        let hx = TimeDepMatrixTerm::new(&sx2, &fx);
        let hz = TimeDepMatrixTerm::new(&sz2, &fz);
        let haml = TimeDepMatrix{terms: vec![hx, hz]};

        let h1 = haml.eval(0.5);
        println!("t=0.5, h=\n{}", h1);

    }
}
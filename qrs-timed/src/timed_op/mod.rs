use qrs_core::quantum::{QObj, QRep};
use qrs_core::{ComplexScalar};
use qrs_core::{ComplexField,RealField, RealScalar};
use std::borrow::Cow;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct TimeDependentOperatorTerm<
    'a, R: RealField, N: ComplexScalar,
    Q: QRep<N>, T: QObj<N, Rep=Q>>
{
    pub op: Cow<'a, T>,
    pub f: &'a dyn Fn(R)->N,
    _phantom: PhantomData<Q>
}

#[derive(Clone)]
pub struct TimeDependentOperator<'a, R: RealField, N: ComplexScalar,
    Q: QRep<N>, T: QObj<N, Rep=Q>>{
    pub terms: Vec<TimeDependentOperatorTerm<'a, R, N, Q, T>>,
}

impl<'a, R: RealScalar, N: ComplexScalar<R=R>, Q: QRep<N>, T: QObj<N, Rep=Q>>
TimeDependentOperator<'a, R, N, Q, T>{
    pub fn new(terms: Vec<TimeDependentOperatorTerm<'a, R, N, Q, T>>) -> Self{
        Self{terms}
    }
}

impl<'a, R: RealField, N: ComplexScalar, Q: QRep<N>, T: QObj<N, Rep=Q>>
TimeDependentOperatorTerm<'a, R, N, Q, T>
    where N: ComplexField{
    pub fn new(q: &'a T, f: &'a dyn Fn(R) -> N) -> Self{
        Self{op: Cow::Borrowed(q), f, _phantom: Default::default() }
    }
    pub fn new_with_owned(q: T, f: &'a dyn Fn(R) -> N) -> Self{
        Self{op: Cow::Owned(q), f, _phantom: Default::default() }
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
        use super::{TimeDependentOperatorTerm, TimeDependentOperator};
        use qrs_core::ComplexField;
        use num_complex::Complex64 as c64;
        //use crate::base::DenseQRep;
        use qrs_core::quantum::{TensorProd};
        use qrs_core::reps::matrix::{Op};

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
        let fz = |s: f64| c64::from_real(s);
        let hx = TimeDependentOperatorTerm::new(&sx2, &fx);
        let hz = TimeDependentOperatorTerm::new(&sz2, &fz);
        let haml = TimeDependentOperator{terms: vec![hx, hz]};

        let h1 = haml.eval(0.5);
        println!("t=0.5, h=\n{}", h1);

    }
}
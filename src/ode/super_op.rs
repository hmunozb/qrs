use alga::general::{RealField, ComplexField};
use expm::Expm;
use ndarray::Array2;
use ndarray::{ArrayView, ArrayView2};
use ndarray::ShapeBuilder;
use num_complex::Complex;
use num_traits::Float;
use vec_ode::exp::{ExponentialSplit, Commutator};

use crate::base::dense::*;
use blas_traits::BlasScalar;
use crate::util::{EigResolver, EigJob, EigRangeData, outer_zip_to, change_basis_to, change_basis, unchange_basis};
use std::iter::FromIterator;
use nalgebra::{DMatrix, Dynamic, Matrix};
use itertools::Itertools;

///Defines the exponential e^{-i H} for a Hermitian operator H
/// For Split ODE solvers
pub struct DenMatExpiSplit<T>
    where Complex<T> : BlasScalar
{
    n: usize,
    eiger: EigResolver<Complex<T>>
}

impl<T> DenMatExpiSplit<T>
    where Complex<T> : BlasScalar
{
    pub fn new(n: u32) -> Self{
        Self{n: n as usize, eiger: EigResolver::new_eiger(n, EigJob::ValsVecs, EigRangeData::all())}
    }
}


impl<T: RealField+Float> ExponentialSplit<T, Complex<T>, Op<T>> for DenMatExpiSplit<T>
    where Complex<T> : BlasScalar + ComplexField<RealField=T>
{
    type L = Op<T>;
    type U = (Op<T>, Op<T>);

    fn lin_zero(&self) -> Op<T> {
        Op::zeros(self.n, self.n)
    }

    fn exp(&mut self, l: &Op<T>) -> (Op<T>, Op<T>){
        self.eiger.borrow_matrix().copy_from(l);
        self.eiger.eig();

        let (vals, vecs ) = (self.eiger.vals().clone(),
                             self. eiger.vecs().clone());
        let mut freqs : DMatrix<T> = DMatrix::zeros(self.n, self.n);
        outer_zip_to(&vals, &vals, & mut freqs, |a, b| *b - *a);

        let expifreqs: Vec<Complex<T>> = Vec::from_iter(
            freqs.into_iter().map(|v|
                Complex::exp(&(-Complex::i() * Complex::from(v))) ));

        let expifreqs = Op::from_column_slice(self.n, self.n, &expifreqs);

        (expifreqs, vecs)
    }

    fn map_exp(&mut self, u: & Self::U, x: & Op<T>) -> Op<T>{
        let mut y = change_basis(x, &u.1);
        y.component_mul_assign(&u.0);
        unchange_basis( &y , &u.1)
    }

    fn multi_exp(&mut self, l: &Op<T>, k_arr: &[Complex<T>]) -> Vec<Self::U>{
        self.eiger.borrow_matrix().copy_from(l);
        self.eiger.eig();

        let (vals, vecs ) = (self.eiger.vals().clone(),
                             self. eiger.vecs().clone());
        let mut freqs : DMatrix<T> = DMatrix::zeros(self.n, self.n);
        outer_zip_to(&vals, &vals, & mut freqs, |a, b| *b - *a);

        let mut u_vec = Vec::new();

        for &k in k_arr{
            let expifreqs : Vec<Complex<T>> = freqs.iter()
                .map(|v|  Complex::exp(&(-Complex::i() * k * Complex::from(v))))
                .collect_vec();
            let expifreqs_arr = Op::from_column_slice(self.n, self.n, &expifreqs);

            u_vec.push((expifreqs_arr, vecs.clone()))
        }

        u_vec
    }
}

pub struct DenMatPerturbExpSplit<T: RealField>
    where Complex<T> : BlasScalar{
    n: usize,
    expm: Expm<Complex<T>>
}
impl<T: RealField> DenMatPerturbExpSplit<T>
where Complex<T> : BlasScalar{
    pub fn new(n: u32) -> Self{
        Self{n: n as usize, expm: Expm::new(n as usize )}
    }
}
impl<T: RealField> ExponentialSplit<T, Complex<T>, Op<T>> for DenMatPerturbExpSplit<T>
where Complex<T> : BlasScalar
{
    type L = Op<T>;
    type U = Op<T>;

    fn lin_zero(&self) -> Self::L {
        Op::zeros(self.n, self.n)
    }

    fn exp(&mut self, l: &Self::L) -> Self::U {
        let n = self.n;
        let arr = ArrayView2::from_shape((n,n), & l.as_slice()).unwrap();
        let mut exp_arr = Array2::zeros((n,n));
        self.expm.expm(&arr, &mut exp_arr);
        let exp_map: Op<T> = Op::from_vec(n, n, exp_arr.into_owned().into_raw_vec());

        exp_map
    }

    fn map_exp(&mut self, u: &Self::U, x: &Op<T>) -> Op<T> {
        //u.ad_mul(x) * u
        u * (x * &u.adjoint())
    }
}


impl<T: RealField+Float> Commutator<T, Complex<T>, Op<T>> for DenMatExpiSplit<T>
    where Complex<T> : BlasScalar + ComplexField<RealField=T>{
    /// The expi bracket is defined as
    /// [LA, LB] = -i[A, B]
    /// consistent with the Lie algebra generated by LA = -i A and LB = -i B
    ///
    /// exp (a LA + b LB + c[LA, LB] + ...)
    /// exp ( a (-i A) + b (-i B) - c [A ,B] + ...)
    /// = expi( a A + b B - c i [A, B] + ...)
    fn commutator(&self, la: &Op<T>, lb: &Op<T>) -> Op<T>{
        let mut c : Op<T> = la * lb - lb * la;
        c *= -Complex::i();
        c
    }
}

/// Defines an exponential split for off-diagonal n x n matrices A
/// where exp(A) is the *componentwise* exponent of A
/// and the action of exp(A) on a density matrix rho is componentwise multiplication
pub struct CoherentExpSplit{
    n: usize
}

impl CoherentExpSplit{
    pub fn new(n: u32) -> Self{
        Self{n: n as usize}
    }
}

impl<T: RealField+Float> ExponentialSplit<T, Complex<T>, Op<T>> for CoherentExpSplit{
    type L = Op<T>;
    type U = Op<T>;

    fn lin_zero(&self) -> Self::L {
        Op::zeros(self.n, self.n)
    }

    fn exp(&mut self, l: &Op<T>) -> Op<T>{
        l.map( |x| x.exp() )
    }

    fn map_exp(&mut self, u: & Op<T>, x: &Op<T>) -> Op<T>{
        u.component_mul(x)
    }
}

/// Defines an exponential split for transition rate matrices A.
/// A population conserving transition matrix has the properties
///   1. -A_{ii} \geq 0
///   2. A_{ij} \geq 0,  j\neq i
///   3. \sum_{j} A_{ij} = 0
/// If p is a population vector, then A defines a Markovian kinetic master equation
///  \dv{p}{t} = A p
///
pub struct KineticExpSplit<T: RealField>
where Complex<T>: BlasScalar
{
    n: usize,
    expm: Expm<Complex<T>>

}

impl<T: RealField> KineticExpSplit<T>
where Complex<T>: BlasScalar
{
    pub fn new(n : u32) -> Self{
        Self{n: n as usize, expm: Expm::new(n as usize )}
    }
}

impl<T: RealField> ExponentialSplit<T, Complex<T>, Op<T>> for KineticExpSplit<T>
where Complex<T>: BlasScalar
{
    type L = Op<T>;
    type U = Op<T>;

    fn lin_zero(&self) -> Self::L {
        Op::zeros(self.n, self.n)
    }

    fn exp(&mut self, l: &Op<T>) -> Op<T>{
        let n = self.n;
        //let re_l = l.map(|x| x.re.to_f64().unwrap() );
        // Construct a array from a *column major* slice
        let arr = ArrayView2::from_shape((n,n), & l.as_slice()).unwrap();
        let mut exp_arr = Array2::zeros((n,n));
        //let mut exp_mat: Op<T> = Op::zeros(n, n);
        self.expm.expm(&arr, &mut exp_arr);

        let exp_map: Op<T> = Op::from_vec(n, n, exp_arr.into_owned().into_raw_vec());
//        for (a, b) in exp_mat.iter_mut()
//                .zip(exp_arr.into_iter()){
//            *a = Complex::from( T::from_subset( b))
//        }

        exp_map
    }

    fn map_exp(&mut self, u: & Op<T>, x: &Op<T>) -> Op<T>{
        let n = self.n;
        let mut v : Ket<T> = Ket::zeros(n);
        let mut y = x.clone();

        for i in 0..n{ //diagonal iteration
            let k = (n+1)*i;
            v[i] = *x.get(k).unwrap();
        }

        let w = u * v;
        for i in 0..n{ //diagonal iteration
            let k = (n+1)*i;
            y[k] = w[i];
        }

        y
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use openblas_src;
    use lapacke_sys;
    use num_complex::Complex64 as c64;
    #[test]
    fn super_op_splits(){
        let _0 = c64::from(0.0);
        let _1 = c64::from(-1.0);
        let mut ksplit = KineticExpSplit::new(3);
        let m1 : Op<f64> = Op::from_row_slice(3, 3,
                                              &[ -_1, _1, _0,
                                                c64::from(0.25), c64::from(-0.25), _0,
                                                _0, _0, _0]);
        let x1 : Op<f64> = Op::from_row_slice(3, 3,
                                              &[_1, _0, _0,
                                                  _0, c64::from(0.5), _0,
                                                  _0, _0, c64::from(1.0) ]);
        let u1 = ksplit.exp(&m1);
        let y1 = ksplit.map_exp(&u1, &x1);

        println!(" Rates:\n{}\n\n Exponential:\n{}\n", m1, u1);
        println!(" x1:\n{}\n\n y1:\n{}\n", x1, y1);

    }
}
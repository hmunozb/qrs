use cblas::{Layout, Transpose};
use itertools::Itertools;
use lapack_traits::LapackScalar;
use ndarray::prelude::*;
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use simba::scalar::ClosedMul;

use crate::{ComplexField, RealField};

/// Computes
/// out[i, j] = f( a[i], b[j] )
pub fn outer_zip_to<N, N2, F>
(
    a: &Array1<N>,
    b: &Array1<N>,
    out: &mut Array2<N2>,
    f: F
)
    where
          F: Fn(&N, &N) -> N2{

    let sh = out.shape();
    assert!(sh[0] == a.len() && sh[1] == b.len(),
            "outer_zip_to Dimensions mismatch");

    for(m, (cb, ca)) in out.iter_mut()
        .zip( // row major iterator
              a.iter().cartesian_product(b.iter()) ){
        *m = f(ca, cb);
    }
}

/// Evaluates the kronecker product of the two arrays
pub fn kronecker<N: Copy + ClosedMul>(a: ArrayView2<N>, b: ArrayView2<N>) -> Array2<N>{
    let a_sh = a.raw_dim();
    let b_sh = b.raw_dim();
    let c_sh = (a_sh[0] * b_sh[0], a_sh[1] * b_sh[1]);

    // index iteration will initalize all elements
    let mut c = Array2::uninit(c_sh);
    //let mut c : Array2<N> = unsafe{Array2::uninitialized(c_sh)};
    for ((i,j), aij) in a.indexed_iter(){
        for ((k, l), bkl) in b.indexed_iter(){
            unsafe {
                c.uget_mut((i*b_sh[0] + k, j*b_sh[1] + l))
                    .write(*aij * *bkl);
            }
        }
    }
    let c = unsafe { c.assume_init()};
    c
}

pub fn copy_to_complex<N>(
    a: &Array2<N>,
    b: &mut Array2<Complex<N>>
)
    where N: RealField + Float,
{
    assert_eq!(a.shape(), b.shape(), "copy_transmute_to Dimensions mismatch");

    for (&ca, cb) in a.iter().zip(b.iter_mut()){
        *cb = Complex::new(ca, N::zero());
    }
}

pub fn gemm<N: Zero+One+LapackScalar>(
    c: &mut Array2<N>, a: ArrayView2<N>, b: ArrayView2<N>, a_t: Transpose, b_t: Transpose ){
    let mut a_sh = a.raw_dim();
    let mut b_sh = b.raw_dim();
    let mut c_sh = c.raw_dim();
    //let al = a.is_standard_layout();
    //let bl = b.is_standard_layout();

    let at_sh = if let Transpose::None = a_t {a_sh} else { a_sh.slice_mut().reverse(); a_sh};
    let bt_sh = if let Transpose::None = b_t {b_sh} else { a_sh.slice_mut().reverse(); b_sh};
    assert_eq!(at_sh[1], bt_sh[0], "Shape mismatch: A = {:?}, B = {:?}", at_sh, bt_sh);
    assert_eq!(at_sh[0], c_sh[0], "Shape mismatch: A = {:?}, C = {:?}", at_sh, c_sh);
    assert_eq!(bt_sh[1], c_sh[1], "Shape mismatch: B = {:?}, C = {:?}", bt_sh, c_sh);

    unsafe{
        N::gemm(Layout::RowMajor, a_t, b_t,
            at_sh[0] as i32, bt_sh[1] as i32,
            at_sh[1] as i32, N::one(),
            a.as_slice().unwrap(),  a_sh[0] as i32,
            b.as_slice().unwrap(), b_sh[0] as i32, N::zero(),
            c.as_slice_mut().unwrap(), c_sh[0] as i32)
    };
}

pub fn ad_mul_to<N: Zero+One+LapackScalar>(
    a: ArrayView2<N>, b: ArrayView2<N>, c: &mut Array2<N>
){
    gemm(c, a, b, Transpose::Conjugate, Transpose::None);
}

#[allow(non_snake_case)]
/// Z <- U^dag A U
pub fn change_basis_to<N: Zero+One+LapackScalar>(
    A: ArrayView2<N>,
    U: ArrayView2<N>,
    //_temp: &mut Array2<N>,
    out: &mut Array2<N>){
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : Array2<N> = Array2::zeros((a_sh[0], u_sh[1]));
    gemm(&mut m1, A, U, Transpose::None, Transpose::None);
    gemm(out, U, m1.view(), Transpose::Conjugate, Transpose::None);
}

#[allow(non_snake_case)]
/// Performs U^dag A U
pub fn change_basis<N: Zero+One+LapackScalar>(
    A: ArrayView2<N>,
    U: ArrayView2<N>) -> Array2<N>{
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : Array2<N> = Array::zeros((a_sh[0], u_sh[1]));
    let mut m2 : Array2<N> = Array::zeros((u_sh[1], u_sh[1]));
    gemm(&mut m1, A, U, Transpose::None, Transpose::None);
    gemm(&mut m2, U, m1.view(), Transpose::Conjugate, Transpose::None);
    m2
    //note: threshold size for zgemm to be efficient over matrixmultiply may be above 16x16
    //U.ad_mul(&(A * U))
}

#[allow(non_snake_case)]
/// Performs U A U^dag where A is Hermitian
pub fn unchange_basis<N: Zero+One+LapackScalar>(
    A: ArrayView2<N>,
    U: ArrayView2<N>) -> Array2<N>{
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : Array2<N> = Array2::zeros((a_sh[0], u_sh[0]));
    let mut m2 : Array2<N> = Array2::zeros((u_sh[0], u_sh[0]));
    gemm(&mut m1, A, U, Transpose::None, Transpose::Conjugate);
    gemm(&mut m2, U, m1.view(), Transpose::None, Transpose::None);
    m2
    // U * ( A * U.adjoint())
}

/// Adds the sum of the matrices in arr to out
pub fn reduce_sum<N: ComplexField>(
    arr: &Vec<Array2<N>>,
    out: &mut Array2<N>
)
{
    for a in arr.iter(){
        *out += a;
    }
}
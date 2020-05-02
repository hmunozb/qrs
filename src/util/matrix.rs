use nalgebra::{Matrix,Vector,DMatrix};
use nalgebra::{Scalar, Dim, U1};
use nalgebra::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
use itertools::Itertools;
use crate::{ComplexField, RealField};
use num_complex::Complex;
use cblas::{Transpose, Layout};
use lapack_traits::LapackScalar;

/// Computes
/// out[i, j] = f( a[i], b[j] )
pub fn outer_zip_to<N, D1, S1, R2, C2, S2, F>
(
    a: &Vector<N, D1, S1>,
    b: &Vector<N, D1, S1>,
    out: &mut Matrix<N, R2, C2, S2>,
    f: F
)
where N: Scalar, D1: Dim, S1: ContiguousStorage<N, D1, U1>,
R2: Dim, C2: Dim, S2: ContiguousStorageMut<N, R2, C2>,
F: Fn(&N, &N) -> N{

    let (nrows, ncols) = out.shape();
    assert!(nrows == a.len() && ncols == b.len(),
            "outer_zip_to Dimensions mismatch");

    for(m, (cb, ca)) in out.as_mut_slice().iter_mut().zip( // Column major iterator
    b.as_slice().iter().cartesian_product(a.as_slice().iter()) ){
        *m = f(ca, cb);
    }
}

pub fn copy_transmute_to<N, R, C, S1, S2>(
    a: &Matrix<N, R, C, S1>,
    b: &mut Matrix<Complex<N>, R, C, S2>
)
where N:Scalar+RealField, R: Dim, C: Dim, S1: Storage<N, R, C>, S2: StorageMut<Complex<N>, R, C>
{
    assert!(a.shape() == b.shape(), "copy_transmute_to Dimensions mismatch");

    for (ca, cb) in a.iter().zip(b.iter_mut()){
        *cb = Complex::from_real(*ca);
    }
}

pub fn gemm<N: Scalar + LapackScalar>(
        c: &mut DMatrix<N>, a: &DMatrix<N>, b: &DMatrix<N>, a_t: Transpose, b_t: Transpose ){
    let a_sh = a.shape();
    let b_sh = b.shape();
    let c_sh = c.shape();
    let at_sh = if let Transpose::None = a_t {a_sh} else { (a_sh.1, a_sh.0)};
    let bt_sh = if let Transpose::None = b_t {b_sh} else { (b_sh.1, b_sh.0)};
    assert!(at_sh.1 == bt_sh.0, format!("Shape mismatch: A = {:?}, B = {:?}", at_sh, bt_sh));
    assert!(at_sh.0 == c_sh.0, format!("Shape mismatch: A = {:?}, C = {:?}", at_sh, c_sh));
    assert!(bt_sh.1 == c_sh.1, format!("Shape mismatch: B = {:?}, C = {:?}", bt_sh, c_sh));

    unsafe {
        N::gemm(Layout::ColumnMajor, a_t, b_t,
                at_sh.0 as i32, bt_sh.1 as i32,
                at_sh.1 as i32, N::one(), a.as_slice(), a_sh.0 as i32, b.as_slice(),
                b_sh.0 as i32, N::zero(), c.as_mut_slice(), c_sh.0 as i32)
    };
}

pub fn ad_mul_to<N: Scalar + LapackScalar>(
    a: &DMatrix<N>, b: &DMatrix<N>, c: &mut DMatrix<N>
){
    gemm(c, a, b, Transpose::Conjugate, Transpose::None);
}

#[allow(non_snake_case)]
/// Z <- U^dag A U
pub fn change_basis_to<N: Scalar+LapackScalar>(
        A: &DMatrix<N>,
        U: &DMatrix<N>,
        _temp: &mut DMatrix<N>,
        out: &mut DMatrix<N>){
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : DMatrix<N> = DMatrix::zeros(a_sh.0, u_sh.1);
    gemm(&mut m1, A, U, Transpose::None, Transpose::None);
    gemm(out, U, &m1, Transpose::Conjugate, Transpose::None);
}

#[allow(non_snake_case)]
/// Performs U^dag A U
pub fn change_basis<N: Scalar+LapackScalar>(
    A: &DMatrix<N>,
    U: &DMatrix<N>) -> DMatrix<N>{
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : DMatrix<N> = DMatrix::zeros(a_sh.0, u_sh.1);
    let mut m2 : DMatrix<N> = DMatrix::zeros(u_sh.1, u_sh.1);
    gemm(&mut m1, A, U, Transpose::None, Transpose::None);
    gemm(&mut m2, U, &m1, Transpose::Conjugate, Transpose::None);
    m2
    //note: threshold size for zgemm to be efficient over matrixmultiply may be above 16x16
    //U.ad_mul(&(A * U))
}

#[allow(non_snake_case)]
/// Performs U A U^dag where A is Hermitian
pub fn unchange_basis<N: Scalar+LapackScalar>(
    A: &DMatrix<N>,
    U: &DMatrix<N>) -> DMatrix<N>{
    let a_sh = A.shape();
    let u_sh = U.shape();
    let mut m1 : DMatrix<N> = DMatrix::zeros(a_sh.0, u_sh.0);
    let mut m2 : DMatrix<N> = DMatrix::zeros(u_sh.0, u_sh.0);
    gemm(&mut m1, A, U, Transpose::None, Transpose::Conjugate);
    gemm(&mut m2, U, &m1, Transpose::None, Transpose::None);
    m2
    // U * ( A * U.adjoint())
}

/// Adds the sum of the matrices in arr to out
pub fn reduce_sum<N: Scalar+ComplexField, R, C,S1, S2>(
    arr: &Vec<Matrix<N, R, C, S1>>,
    out: &mut Matrix<N, R, C, S2>
)
where R: Dim, C: Dim, S1: Storage<N, R, C>, S2: StorageMut<N, R, C>
{
    for a in arr.iter(){
        *out += a;
    }
}
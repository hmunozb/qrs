use nalgebra::{Matrix,Vector,DMatrix};
use nalgebra::{Scalar, Dim, U1};
use nalgebra::base::storage::{ContiguousStorage, ContiguousStorageMut, Storage, StorageMut};
use itertools::Itertools;
use alga::general::{ComplexField, RealField};
use num_complex::Complex;

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

/// Z <- U^dag A U
pub fn change_basis_to<N: Scalar+ComplexField>(
        A: &DMatrix<N>,
        U: &DMatrix<N>,
        temp: &mut DMatrix<N>,
        out: &mut DMatrix<N>){
    A.mul_to(U, temp);
    U.ad_mul_to(temp, out);
}

/// Performs U^dag A U
pub fn change_basis<N: Scalar+ComplexField>(
    A: &DMatrix<N>,
    U: &DMatrix<N>) -> DMatrix<N>{
    U.ad_mul(&(A * U))
}

/// Performs U A U^dag where A is Hermitian
pub fn unchange_basis<N: Scalar+ComplexField>(
    A: &DMatrix<N>,
    U: &DMatrix<N>) -> DMatrix<N>{
    U * ( A * U.adjoint())
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
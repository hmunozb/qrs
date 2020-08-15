#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use criterion::{BenchmarkId, Criterion};
use nalgebra::DMatrix;
use num_complex::Complex64 as c64;
use rand::prelude::*;
use rand_distr::{Normal};

use qrs_core::reps::matrix::*;
use qrs_core::eig::dense::EigResolver;
use qrs_core::eig::{EigJob, Layout, QEiger};
use qrs::util::*;


static _1z: c64 = c64{re:1.0, im:0.0};
static _m1z: c64 = c64{re:-1.0, im:0.0};
static _0z: c64 = c64{re:0.0, im:0.0};
static _Iz: c64 = c64{re:0.0, im:1.0};
static _mIz: c64 = c64{re:0.0, im:-1.0};

static _sx : [c64; 4] = [_0z, _1z,
    _1z, _0z];
static _sy : [c64; 4] = [_0z, _Iz,
    _mIz, _0z]; //column major matrix
static _sz : [c64; 4] = [   _1z, _0z,
        _0z, _m1z];

#[test]
fn test_hermitian(){
    use qrs_core::quantum::QObj;
    use qrs::base::pauli::matrix as pauli;

    let sx = pauli::sx::<f64>();
    let sy = pauli::sy::<f64>();
    let sz = pauli::sz::<f64>();

    let h =  sx.clone().qscal(Complex::from(0.5)) + sy;
    println!("Matrix:\n {}", h);

    let mut eig_resolver : EigResolver<c64> = EigResolver::new(
        2, EigJob::ValsVecs, EigRangeData::all(),  Layout::RowMajor,false);

    let m = eig_resolver.borrow_matrix();
    m.copy_from(& h);
    //m[(1, 0)] = _0z;
    println!("C Upper Triangular layout:\n{:#?}", m.as_slice());

    let (vals, vecs) = eig_resolver.eigh(&h);
    let vals = DMatrix::from_shape_vec(2, vals);
    println!("Eigenvalues:\n {} ", vals);
    println!("Eigenvectors:\n {} ", vecs);

    println!("Working!...");

    for i in 0..100{
        let m = eig_resolver.borrow_matrix();
        m.copy_from(& h);
        eig_resolver.eig();
    }
}

fn rand_herm<Fun>(n: u32, k: u32, f: &mut Fun)
where Fun: FnMut(&mut DMatrix<c64>)
{
    //assert!(h.is_square());
    //let n = h.nrows() as u32;

    let mut eig_resolver : EigResolver<c64> = EigResolver::new(
        n, EigJob::ValsVecs, EigRangeData::all(),  Layout::RowMajor,false);

    let mut m = Op::zeros(n as usize, n as usize);
    for _i in 0..k {
        f(&mut m);
        QEiger::<c64, DenseQRep<c64>>::eigh(&mut eig_resolver, &m);
    }

}

pub fn bench_random_10x10_hermitian(c: &mut Criterion){
    bench_random_hermitian(c, 10);
}

pub fn bench_random_32x32_hermitian(c: &mut Criterion){
    bench_random_hermitian(c, 32);
}

pub fn bench_random_64x64_hermitian(c: &mut Criterion){
    bench_random_hermitian(c, 64);
}

pub fn bench_random_256x256_hermitian(c: &mut Criterion){
    bench_random_hermitian(c, 256);
}

fn bench_random_hermitian(c: &mut Criterion, n: usize){
    let group_name = format!("Eig Resolver: Random Hermitian {0}x{0}", n);
    let mut group = c.benchmark_group(group_name);


    let mut r = StdRng::from_rng(thread_rng()).unwrap();
    let dist = Normal::new(0.0,  0.5_f64.sqrt()).unwrap();

    let mut rand_normal_c64 = || c64::new(r.sample(dist), r.sample(dist));
    let mut rh : DMatrix<c64> = DMatrix::from_fn(n, n, |_i, _j| { rand_normal_c64() });
    let mut reg: DMatrix<c64> = DMatrix::zeros(n,n);

    //Generates a random positive definite matrix with unit trace
    let mut f = |m: &mut DMatrix<c64>| {
        for ci in rh.as_mut_slice().iter_mut(){
            *ci = rand_normal_c64();
        }
        rh.adjoint_to(&mut reg);
        reg += & rh;
        reg.scale_mut(2.0);
        reg.adjoint_to(&mut rh);
        rh.mul_to(&reg, m);

        let trm: c64 = m.trace();
        m.unscale_mut(trm.re);
    };

    for i in [ 10, 25, 50, 100].iter(){

//        let mut rh : DMatrix<c64> = DMatrix::from_fn(n, n, |i, j| { rand_normal_c64() });
//        let h :DMatrix<c64> = rh.hermitian_part();

        group.bench_with_input(BenchmarkId::new("EigResolver Reused", i),
            i,
            |b,j| b.iter(
                ||{
                    rand_herm(n as u32, *j as u32, &mut f)})
        );

        group.bench_with_input(BenchmarkId::new("EigResolver Reallocated", i),
               i,
               |b,j| b.iter(
                   ||{
                       for _ in 0..*j{ rand_herm(n as u32, 1, &mut f) } })
        );
    }
//
//    c.bench_function("10x10 Loop 50",|b|{
//        b.iter(||{ rand_herm(&h10, 50) } ) } );
//
//    c.bench_function("10x10 Loop 1, Do 50",|b|{
//        b.iter(||{ for i in 0..50 {rand_herm(&h10, 1);} } ) } );
}
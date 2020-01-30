//! Support for simd/packed floating point operations
//! (The alternative packed_simd crate is not yet available on the stable channel)
//use std::mem;
use std::ops::{Add, AddAssign, Mul, MulAssign, SubAssign, Sub, Neg};
use num_traits::{Zero, Float};
//use itertools::Itertools;
//use packed_simd::f64x4;

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C, align(32))]
pub struct Aligned4xf64{
    pub dat: [f64; 4]
}

impl Aligned4xf64{
    fn map<F>(&self, f: F) -> Aligned4xf64
    where F: Fn(f64) -> f64{
        let mut c = Aligned4xf64::default();
        for (c, &a) in c.dat.iter_mut().zip(self.dat.iter()){
            *c = f(a);
        }
        c
    }
}

impl From<[f64; 4]> for Aligned4xf64{
    fn from(dat: [f64; 4]) -> Self {
        Self{dat}
    }
}

impl Default for Aligned4xf64{
    fn default() -> Self {
        Self::zero()
    }
}
impl Zero for Aligned4xf64{
    fn zero() -> Self {
        Self{dat: [0.0, 0.0, 0.0, 0.0]}
    }

    fn is_zero(&self) -> bool {
        self.dat.iter().all(|a| a.is_zero())
    }
}
impl Add for Aligned4xf64{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut me = self;
        me += rhs;
        me
    }
}
impl AddAssign<f64> for Aligned4xf64{
    #[allow(unreachable_code)]
    fn add_assign(&mut self, rhs: f64) {
        let mut me = self;
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(me.dat.as_ptr());
            let mmb = x86_64::_mm256_broadcast_sd(&rhs);
            let mm_sum = x86_64::_mm256_add_pd(mma, mmb);
            x86_64::_mm256_store_pd(me.dat.as_mut_ptr(), mm_sum);
            return;
        }

        for a in me.dat.iter_mut(){
            *a += rhs;
        }
    }
}
impl Add<f64> for Aligned4xf64{
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        let mut me = self;
        me += rhs;
        me
    }
}
impl SubAssign for Aligned4xf64{
    #[allow(unreachable_code)]
    fn sub_assign(&mut self, rhs: Self) {
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(self.dat.as_ptr());
            let mmb = x86_64::_mm256_load_pd(rhs.dat.as_ptr());
            let mm_neg = x86_64::_mm256_sub_pd(mma, mmb);
            x86_64::_mm256_store_pd(self.dat.as_mut_ptr(), mm_neg);
            return;
        }

        for (a, &b) in self.dat.iter_mut().zip(rhs.dat.iter()){
            *a -= b;
        }
    }
}
impl Sub for Aligned4xf64{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut me = self;
        me -= rhs;
        me
    }
}
impl AddAssign for Aligned4xf64{
    #[allow(unreachable_code)]
    fn add_assign(&mut self, rhs: Self) {
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(self.dat.as_ptr());
            let mmb = x86_64::_mm256_load_pd(rhs.dat.as_ptr());
            let mm_sum = x86_64::_mm256_add_pd(mma, mmb);
            x86_64::_mm256_store_pd(self.dat.as_mut_ptr(), mm_sum);
            return;
        }
        for (a, &b) in self.dat.iter_mut().zip(rhs.dat.iter()){
            *a += b
        }
    }
}
impl MulAssign<f64> for Aligned4xf64{
    #[allow(unreachable_code)]
    fn mul_assign(&mut self, rhs: f64) {
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(self.dat.as_ptr());
            let mmb = x86_64::_mm256_broadcast_sd(&rhs);
            let mm_mul = x86_64::_mm256_mul_pd(mma, mmb);
            x86_64::_mm256_store_pd(self.dat.as_mut_ptr(), mm_mul);
            return;
        }

        for a in self.dat.iter_mut(){ *a *= rhs} ;
    }
}
impl MulAssign<Aligned4xf64> for Aligned4xf64{
    #[allow(unreachable_code)]
    fn mul_assign(&mut self, rhs: Aligned4xf64) {
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(self.dat.as_ptr());
            let mmb = x86_64::_mm256_load_pd(rhs.dat.as_ptr());
            let mm_mul = x86_64::_mm256_mul_pd(mma, mmb);
            x86_64::_mm256_store_pd(self.dat.as_mut_ptr(), mm_mul);
            return;
        }

        for (a, b) in self.dat.iter_mut().zip(rhs.dat.iter()){
            *a *= b;
        }
    }
}
impl Mul<Aligned4xf64> for Aligned4xf64{
    type Output = Self;

    fn mul(self, rhs: Aligned4xf64) -> Self::Output {
        let mut me = self;
        me *= rhs;
        me
    }
}
impl Mul<f64> for Aligned4xf64{
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        let mut me = self;
        me *= rhs;
        me
    }
}
impl Neg for Aligned4xf64{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut me = self;
        #[target_feature(enable="avx2")]
        unsafe {
            use std::arch::x86_64;
            let mma = x86_64::_mm256_load_pd(me.dat.as_ptr());
            let mmb =x86_64::_mm256_broadcast_sd(&(-1.0));
            let mm_neg = x86_64::_mm256_mul_pd(mma, mmb);
            x86_64::_mm256_store_pd(me.dat.as_mut_ptr(), mm_neg);
            return me;
        }

        for a in me.dat.iter_mut(){
            *a *= -1.0;
        }
        me
    }
}
//pub fn aligned_vector_f64(n: usize) -> Vec<f64>{
//    let mut v = Vec::<Aligned4xf64>::with_capacity(n);
//    let p = v.as_mut_ptr();
//    mem::forget(v);
//    unsafe{
//        let p2 : *mut f64= mem::transmute(p);
//        let v = Vec::from_raw_parts(p2, 4*n, 4*n);
//        v
//    }
//}

//pub fn cross_product_simd_4xf64(
//    a: f64x4, b: f64x4
//){
//    let a1 = shuffle!(a, [0, 2, 3, 1]);
//    let b1 = shuffle!(b, [0, 3, 1, 2]);
//    let c1 = a1 * b1;
//    let a2 : f64x4 = shuffle!(a, [0, 3, 1, 2]);
//    let b2 : f64x4 = shuffle!(b, [0, 2, 3, 1]);
//    let c2 = a2 * b2;
//}

pub fn cross_product_aligned_4xf64(
    a: &Aligned4xf64,
    b: &Aligned4xf64,
    c: &mut Aligned4xf64
){
    unsafe{
        cross_product_aligned_f64_arr(&a.dat, &b.dat, &mut c.dat);
    };
}

//#[allow(unreachable_code)]
//pub unsafe fn rotation_matrix_aligned_f64_arr(
//    a: &[f64 ; 4],
//    m: &mut [f64; 12]
//){
//    let mut outer_a : [f64; 12] = [0.0; 12];
//    let mut cross_a : [f64; 12] = [0.0; 12];
//
//    let mut outer_a_chunks = outer_a.chunks_exact_mut(4);
//    let mut cross_a_chunks = cross_a.chunks_exact_mut(4);
//    #[cfg(all(target_arch="x86_64"))]
//    {
//        use std::arch::x86_64;
//        let mma = x86_64::_mm256_load_pd(a.as_ptr());
//
//        let mmoax = x86_64::_mm256_load_pd(outer_a_chunks.next().unwrap().as_ptr());
//        let mmcax = x86_64::_mm256_load_pd(cross_a_chunks.next().unwrap().as_ptr());
//
//        let mma_sq = x86_64::_mm256_mul_pd(mma, mma);
//        let a_sq = x86_64::_mm256_hadd_pd(mma_sq, mma_sq);
//        let a_sq = x86_64::_mm256_hadd_pd(mma_sq, mma_sq);
//
//    }
//}

pub fn vectorized_cross_product_4xf64(
    a: &[Aligned4xf64; 3],
    b: &[Aligned4xf64; 3],
    c: &mut [Aligned4xf64; 3]
){
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/// Analytically evaluate the matrix exponential of the antisymmetric matrix represented
/// in the SO(3) lie algebra basis.
/// Equivalently, the magnitude
pub fn vectorized_3d_cross_exponential_4xf64(
    a: &[Aligned4xf64; 3],
    e: &mut [Aligned4xf64; 9]
){
    let mut omega_mat = *e;
    //let mut omega_mat : [Aligned4xf64; 9] = [Aligned4xf64::zero(); 9];
    let mut omega_sq_mat : [Aligned4xf64; 9] = [Aligned4xf64::zero(); 9];

    let (ax_sq, ay_sq, az_sq) = (a[0]*a[0], a[1]*a[1], a[2]*a[2]);
    let a_sq = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    let mut a_norm = Aligned4xf64::default();
    for (v_norm, &v_sq) in a_norm.dat.iter_mut().zip(a_sq.dat.iter()){
        *v_norm = v_sq.sqrt();
    }
    let rc_a_norm = a_norm.map(|v| 1.0 / v);
    let c_arr = a_norm.map(f64::cos);
    let s_arr = a_norm.map(f64::sin);
    //protect against small-norm division
    let a_norm = a_norm.map(|v| if v.abs() < f64::epsilon(){ 1.0} else { v});

    omega_mat[1] = -a[2] ; omega_mat[2] =  a[1];
    omega_mat[3] =  a[2] ; omega_mat[5] = -a[0];
    omega_mat[6] = -a[1] ; omega_mat[7] =  a[0];

    omega_sq_mat[0] = -(ay_sq + az_sq);  omega_sq_mat[1] = a[0] * a[1];  omega_sq_mat[2] = a[0] * a[2];
    omega_sq_mat[3] = omega_sq_mat[1]; omega_sq_mat[4] = -(ax_sq + az_sq); omega_sq_mat[5] = a[1] * a[2];
    omega_sq_mat[6] = omega_sq_mat[2]; omega_sq_mat[7] = omega_sq_mat[5]; omega_sq_mat[8] = -(ax_sq + ay_sq);


    let sinc_a_norm = s_arr * rc_a_norm;
    for v in omega_mat.iter_mut(){
        *v *= sinc_a_norm;
    }
    let cosc_a_norm =(-c_arr + 1.0) * rc_a_norm * rc_a_norm;
    for v in omega_sq_mat.iter_mut(){
        *v *= cosc_a_norm;
    }

    let mut exp_a = omega_mat;
    exp_a[0] += 1.0; exp_a[4] += 1.0; exp_a[8] += 1.0;
    for (v, w) in exp_a.iter_mut().zip(omega_sq_mat.iter()){
        *v += *w;
    }



}

pub fn vector_cross_product_f64_with_4xf64(
    a: &[f64; 3],
    b: &[Aligned4xf64; 3],
    c: &mut [Aligned4xf64; 3]
){
    c[0] =  b[2] * a[1] - b[1] * a[2];
    c[1] =  b[0] * a[2] - b[2] * a[0];
    c[2] =  b[1] * a[0] - b[0] * a[1] ;
}

#[allow(unreachable_code)]
pub unsafe fn cross_product_aligned_f64_arr(
    a: &[f64 ; 4],
    b: &[f64 ; 4],
    c: &mut [f64; 4]

){
    #[cfg(all(target_arch="x86_64"))]
    {
        use std::arch::x86_64;
        let mma = x86_64::_mm256_load_pd(a.as_ptr());
        let mmb = x86_64::_mm256_load_pd(b.as_ptr());

        let mma1 = x86_64::_mm256_permute4x64_pd(mma, 0x78); // 0 y z x
        let mmb1 = x86_64::_mm256_permute4x64_pd(mmb, 0x9C); // 0 z x y

        let mma2 = x86_64::_mm256_permute4x64_pd(mma, 0x9C);
        let mmb2 = x86_64::_mm256_permute4x64_pd(mmb, 0x78);

        let mmc1 = x86_64::_mm256_mul_pd(mma1, mmb1);
        let mmc2 = x86_64::_mm256_mul_pd(mma2, mmb2);
        let mmc = x86_64::_mm256_sub_pd(mmc1, mmc2);

        x86_64::_mm256_store_pd(c.as_mut_ptr(), mmc);

        return;
    };


    c[1] = a[2] * b[3] - a[3] * b[2];
    c[2] = a[3] * b[1] - a[1] * b[3];
    c[3] = a[1] * b[2] - a[2] * b[1];

}
use crate::{RealScalar, ComplexScalar, SupersetOf};
//use alga::general::{RealScalar, ComplexScalar, SupersetOf};
use lapacke::Layout;
use lapack_traits::{Theevx};
use ndarray::prelude::*;
use num_traits::{Zero, One, ToPrimitive};
use crate::reps::dense::*;
use crate::quantum::eig::{EigRange, EigJob, QEiger, EigQRep};
use crate::quantum::*;

#[derive(Copy,Clone)]
pub struct EigRangeData<F: RealScalar>{
    range: u8,
    vl: F, vu: F,
    il: i32, iu: i32
}

impl<F: RealScalar> EigRangeData<F>{
    pub fn all() -> Self{
        EigRangeData{ range: b'A', vl: F::zero(), vu: F::zero(), il: 0, iu: 0}
    }
    pub fn value_range(vl: F, vu: F) -> Self{
        assert!(vl < vu);
        EigRangeData{ range: b'V', vl, vu, il: 0, iu: 0}
    }

    /// Find the algebraically sorted eigenvalues with indices in the range [il, iu)
    /// where 0 <= il < iu <= n
    /// [ Note: Not the same index convention as a direct syevx/heevx Fortran call
    ///   If you want the k lowest eigenvalues, then (vl, vu) = (0, k) here ]
    pub fn idx_range(il: i32, iu: i32) -> Self{
        assert!(il <= iu);
        EigRangeData{   range: b'I', vl: F::zero(), vu: F::zero(),
            il, iu}
    }
}

impl<F: RealScalar> From<EigRange<F>> for EigRangeData<F>{
    fn from(range: EigRange<F>) -> Self {
        match range{
            EigRange::All => Self::all(),
            EigRange::IdxRange(il, iu) => Self::idx_range(il, iu),
            EigRange::ValRange(vl, vu) => Self::value_range(vl, vu)
        }
    }
}

fn eig_job_u8(job: EigJob) -> u8{
    match job{
        EigJob::ValsVecs => b'V',
        EigJob::ValsOnly => b'N'
    }
}


struct EigWork<N: ComplexScalar>{
    work: Vec<N>,
    rwork: Vec<N::R>,
    iwork: Vec<i32>,
    ifail: Vec<i32>
}

impl<N: ComplexScalar> EigWork<N>{
    fn new() -> Self{
        EigWork{work: Vec::new(), rwork: Vec::new(), iwork: Vec::new(), ifail: Vec::new()}
    }

    fn set_work_sizes(&mut self, lwork: u32, n: u32){
        self.work.resize(lwork as usize, N::zero());
        let rw_size = N::rwork_const() * (n as isize);
        if rw_size <= 0 {
            self.rwork.resize(1, <N::R as Zero>::zero());
        } else {
            self.rwork.resize( rw_size as usize, <N::R as Zero>::zero())
        }

        self.iwork.resize(5 * n as usize, 0);
        self.ifail.resize(n as usize, 0);
    }
}

///
/// EigResolver is a struct that holds on to internal resources required to perform
/// eigendecompositions on hermitian or symmetric (HerSym) matrices
/// This should be utilized when many decompositions are required on many different
/// matrices with the same dimension within a tight loop, rather than calling built-in
/// eig routines that allocate on every single call.
pub struct EigResolver<N: ComplexScalar>{
    jobz: EigJob,
    range: EigRangeData<N::R>,
    a: Array2<N>,
    uplo: u8,
    layout: Layout,
    eigwork: EigWork<N>,
    eigvals: Array1<N::R>,
    eigvecs: Array2<N>
}

impl<N: ComplexScalar> EigResolver<N>
{
    pub fn new_with_raw(n: u32, raw: Vec<N>, jobz: EigJob, range: EigRangeData<N::R>,
               layout: Layout,
               read_upper: bool) -> Self{
        let un = n as usize;
        let num_eigvecs = if range.range == b'I'
        { (range.iu - range.il)  as usize} else { un };

        let a = Array2::from_shape_vec((un, un), raw).unwrap();
        let eigvals = Array1::zeros(un);
        let eigvecs = Array2::zeros((un, num_eigvecs));
        let mut eigwork = EigWork::new();
        eigwork.set_work_sizes(1, n);
        let uplo= if read_upper {
            b'U'  //Upper Fortran <-> Lower C
        } else {
            b'L'  //Lower Fortran <-> Upper C
        };

        let mut me = Self{jobz, range, a, uplo, layout, eigwork, eigvals, eigvecs};
        //Perform a workspace length query
        Self::call_syhe_evx(&mut me, true);
        //This gets stored on the float/complex work array for some reason
        //Hopefully it's a perfectly good positive integer
        let flwork :f64 = me.eigwork.work[0].re().to_f64().unwrap();
        let lwork =  flwork as u32;
        me.eigwork.set_work_sizes(lwork, n);

        return me;

    }
    pub fn new(n: u32, jobz: EigJob, range: EigRangeData<N::R>,
               layout: Layout,
               read_upper: bool) -> Self{
        let un = n as usize;
        let num_eigvecs = if range.range == b'I'
        { (range.iu - range.il)  as usize} else { un };

        let a = Array2::zeros((un, un));
        let eigvals = Array1::zeros(un);
        let eigvecs = Array2::zeros((un, num_eigvecs));
        let mut eigwork = EigWork::new();
        eigwork.set_work_sizes(1, n);
        let uplo= if read_upper {
            b'U'  //Upper Fortran <-> Lower C
        } else {
            b'L'  //Lower Fortran <-> Upper C
        };

        let mut me = Self{jobz, range, a, uplo, layout, eigwork, eigvals, eigvecs};
        //Perform a workspace length query
        Self::call_syhe_evx(&mut me, true);
        //This gets stored on the float/complex work array for some reason
        //Hopefully it's a perfectly good positive integer
        let flwork :f64 = me.eigwork.work[0].re().to_f64().unwrap();
        let lwork =  flwork as u32;
        me.eigwork.set_work_sizes(lwork, n);

        return me;
    }

    /// Constructs a new EigResolver for n x n HerSym matrces
    /// Assumes that the user is working with a data representation that is row major
    /// and that the upper triangle of any HerSym defines its matrix entries
    pub fn new_eiger(n: u32, jobz: EigJob, range: EigRangeData<N::R>) -> Self{
        Self::new(n, jobz, range, Layout::RowMajor,true)
    }

    pub fn take_raw(&mut self, raw: Vec<N>) -> Vec<N>{
        use core::mem::swap;
        let n = self.eigvals.len();
        let mut a = Array2::from_shape_vec((n, n), raw).unwrap();
        swap(&mut a, &mut self.a);

        a.into_raw_vec()
    }
    /// Hands a mutable reference to the internal matrix
    /// The matrix should not be assumed to contain any definite values once eig() is called
    pub fn borrow_matrix(&mut self) -> &mut Array2<N> {
        &mut self.a
    }

    /// Performs the eigendecomposition on the internal matrix
    /// Changing the dimension of the matrix via its &mut is UB
    pub fn eig(&mut self){
        Self::call_syhe_evx(self, false);
    }

    pub fn vals(&self) -> &Array1<N::R>{
        & self.eigvals
    }

    /// Returns a reference to the eigenvector matrix
    /// Each column of the matrix is an eigenvector ordered according to the corresponding eigenvalue
    /// If a (il, iu) range is specify, the number of rows of this matrix is iu - il
    pub fn vecs(&self) -> &Array2<N>{
        & self.eigvecs
    }

    /// Finds the eigenvalues and eigenvectors of the held matrix
    /// and releases the results
    pub fn into_eigs(mut self) -> (Array1<N::R>, Array2<N>){
        self.eig();
        (self.eigvals, self.eigvecs)
    }

    fn call_syhe_evx( er: &mut Self, query: bool){
        let n= er.a.nrows() as i32;
        let a_slice = er.a.as_slice_mut().unwrap();

        let w = er.eigvals.as_slice_mut().unwrap();
        let z = er.eigvecs.as_slice_mut().unwrap();

        let workpad = &mut er.eigwork;

        let lwork : i32 = if query { -1 } else { workpad.work.len() as i32};
        let work =  workpad.work.as_mut_slice();
        let iwork = workpad.iwork.as_mut_slice();
        let rwork = workpad.rwork.as_mut_slice();
        let ifail = workpad.ifail.as_mut_slice();

        let mut m = 0;
        let info = unsafe{
              N::heevx(er.layout,
                    eig_job_u8(er.jobz),
                    er.range.range, er.uplo, n, a_slice, n,
                    er.range.vl.clone(), er.range.vu.clone(),
                    // Different convention for il,iu:
                    // Fortran arrays start at 1, and the range is inclusive
                    er.range.il + 1, er.range.iu,
                    -N::R::one(), &mut m,
                    w, z, n, work, lwork, rwork, iwork, ifail
            )
        };

        if info < 0{
            panic!("Illegal argument error - _syevx/_heevx returned {}", info);
        } else if info > 0{
            panic!("Unexpected computation error - _syevx/_heevx returned {}", info);
        }

    }
}

impl<N: ComplexScalar> QEiger<N, DenseQRep<N>> for EigResolver<N>{
    fn make_eiger(shape: (usize, usize), job: EigJob, range: EigRange<N::R>) -> Self {
        assert_eq!(shape.0, shape.1);

        EigResolver::new(shape.0 as u32, job, EigRangeData::from(range),
                         Layout::RowMajor, true)
    }

    fn eigh(&mut self, op: & Op<N>) -> (Vec<N::R>, Op<N> ){
        self.borrow_matrix().assign(op);
        self.eig();

        (self.vals().clone().into_raw_vec(), self.vecs().clone())
    }

}

impl<N: ComplexScalar> EigQRep<N> for DenseQRep<N>{
    fn eig(op: &Op<N>) -> (Vec<N::R>, Self::OpRep) {
        let mut eiger : EigResolver<N> = QEiger::<N, DenseQRep<N>>::make_eiger(op.qdim(), EigJob::ValsVecs, EigRange::All);
        eiger.borrow_matrix().assign(op);
        eiger.eig();
        let (vals, vecs) = eiger.into_eigs();

        (vals.into_raw_vec(), vecs)
    }
}

#[allow(non_upper_case_globals)]
#[cfg(test)]
mod tests{


}
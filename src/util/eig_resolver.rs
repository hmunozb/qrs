use alga::general::{RealField, ComplexField, SupersetOf};
use lapacke::Layout;
use lapack_traits::Theevx;
use nalgebra::{DMatrix, DVector};
use num_traits::{Zero, One};

#[derive(Copy,Clone)]
pub enum EigJob{
    ValsVecs,
    ValsOnly
}

#[derive(Copy,Clone)]
pub struct EigRangeData<F: RealField>{
    range: u8,
    vl: F, vu: F,
    il: i32, iu: i32
}

impl<F: RealField> EigRangeData<F>{
    pub fn all() -> Self{
        EigRangeData{ range: b'A', vl: F::zero(), vu: F::zero(), il: 0, iu: 0}
    }
    pub fn value_range(vl: F, vu: F) -> Self{
        assert!(vl < vu);
        EigRangeData{ range: b'V', vl, vu, il: 0, iu: 0}
    }

    /// Find the algebraically sorted eigenvalues with indices in the range [il, iu)
    /// where 0 <= il < iu <= n
    /// [ Note: Not the same convention as a direct syevx/heevx Fortran call
    ///   If you want the first k lowest eigenvalues, then (vl, vu) = (0, k) here ]
    pub fn num_range(il: i32, iu: i32) -> Self{
        assert!(il <= iu);
        EigRangeData{   range: b'I', vl: F::zero(), vu: F::zero(),
            il, iu}
    }
}

impl EigJob{
    pub fn val(&self) -> u8{
        match self{
            EigJob::ValsVecs => b'V',
            EigJob::ValsOnly => b'N'
        }
    }
}


struct EigWork<N: ComplexField>{
    work: Vec<N>,
    rwork: Vec<N::RealField>,
    iwork: Vec<i32>,
    ifail: Vec<i32>
}

impl<N: Theevx> EigWork<N>{
    fn new() -> Self{
        EigWork{work: Vec::new(), rwork: Vec::new(), iwork: Vec::new(), ifail: Vec::new()}
    }

    fn set_work_sizes(&mut self, lwork: u32, n: u32){
        self.work.resize(lwork as usize, N::zero());
        let rw_size = N::rwork_const() * (n as isize);
        if rw_size <= 0 {
            self.rwork.resize(1, <N::RealField as Zero>::zero());
        } else {
            self.rwork.resize( rw_size as usize, <N::RealField as Zero>::zero())
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
pub struct EigResolver<N: ComplexField>{
    jobz: EigJob,
    range: EigRangeData<N::RealField>,
    a: DMatrix<N>,
    uplo: u8,
    eigwork: EigWork<N>,
    eigvals: DVector<N::RealField>,
    eigvecs: DMatrix<N>
}

impl<N: Theevx> EigResolver<N>
{
    pub fn new(n: u32, jobz: EigJob, range: EigRangeData<N::RealField>,
               read_upper: bool) -> Self{
        let un = n as usize;
        let num_eigvecs = if range.range == b'I'
        { (range.iu - range.il)  as usize} else { un };

        let a = DMatrix::zeros(un, un);
        let eigvals = DVector::zeros(un);
        let eigvecs = DMatrix::zeros(un, num_eigvecs);
        let mut eigwork = EigWork::new();
        eigwork.set_work_sizes(1, n);
        let uplo;

        if read_upper {
            uplo = b'U';  //Upper Fortran <-> Lower C
        } else {
            uplo = b'L';  //Lower Fortran <-> Upper C
        }

        let mut me = Self{jobz, range, a, uplo, eigwork, eigvals, eigvecs};
        //Perform a workspace length query
        Self::call_syhe_evx(&mut me, true);
        //This gets stored on the float/complex work array for some reason
        //Hopefully it's a perfectly good positive integer
        let flwork :f64 = me.eigwork.work[0].real().to_subset().unwrap();
        let lwork =  flwork as u32;
        me.eigwork.set_work_sizes(lwork, n);

        return me;
    }

    /// Constructs a new EigResolver for n x n HerSym matrces
    /// Assumes that the user is working with a data representation that is row major
    /// and that the upper triangle of any HerSym defines its matrix entries
    pub fn new_eiger(n: u32, jobz: EigJob, range: EigRangeData<N::RealField>) -> Self{
        Self::new(n, jobz, range, true)
    }

    /// Hands a mutable reference to the internal matrix
    /// The matrix should not be assumed to contain any definite values once eig() is called
    pub fn borrow_matrix(&mut self) -> &mut DMatrix<N> {
        &mut self.a
    }

    /// Performs the eigendecomposition on the internal matrix
    /// Changing the dimension of the matrix via its &mut is UB
    pub fn eig(&mut self){
        Self::call_syhe_evx(self, false);

    }

    pub fn vals(&self) -> &DVector<N::RealField>{
        & self.eigvals
    }

    /// Returns a reference to the eigenvector matrix
    /// Each column of the matrix is an eigenvector ordered according to the corresponding eigenvalue
    /// If a (il, iu) range is specify, the number of rows of this matrix is iu - il
    pub fn vecs(&self) -> &DMatrix<N>{
        & self.eigvecs
    }

    /// Finds the eigenvalues and eigenvectors of the held matrix
    /// and releases the results
    pub fn into_eigs(mut self) -> (DVector<N::RealField>, DMatrix<N>){
        self.eig();
        (self.eigvals, self.eigvecs)
    }

    fn call_syhe_evx( er: &mut Self, query: bool){
        let n= er.a.nrows() as i32;
        let a_slice = er.a.as_mut_slice();

        let w = er.eigvals.as_mut_slice();
        let z = er.eigvecs.as_mut_slice();

        let workpad = &mut er.eigwork;

        let lwork : i32 = if query { -1 } else { workpad.work.len() as i32};
        let work =  workpad.work.as_mut_slice();
        let iwork = workpad.iwork.as_mut_slice();
        let rwork = workpad.rwork.as_mut_slice();
        let ifail = workpad.ifail.as_mut_slice();

        let mut m = 0;
        let info = N::heevx(Layout::ColumnMajor,
                                er.jobz.val(),
                               er.range.range, er.uplo, n, a_slice, n,
                               er.range.vl.clone(), er.range.vu.clone(),
                               // Different convention for il,iu:
                               // Fortran arrays start at 1, and the range is inclusive
                               er.range.il + 1, er.range.iu,
                               -N::RealField::one(), &mut m,
                               w, z, n, work, lwork, rwork, iwork, ifail
        );
        if info < 0{
            panic!("Illegal argument error - _syevx/_heevx returned {}", info);
        } else if info > 0{
            panic!("Unexpected computation error - _syevx/_heevx returned {}", info);
        }

    }
}

#[allow(non_upper_case_globals)]
#[cfg(test)]
mod tests{


}
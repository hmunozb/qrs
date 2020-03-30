use nalgebra::{DMatrix, DVector};
use alga::general::{ComplexField, RealField};
use num_complex::Complex;
use smallvec::SmallVec;
use lapack_traits::LapackScalar;
use lapacke::Layout;

pub type DegenArray = Vec<SmallVec<[usize; 4]>>;




/// Friendly reminder that Gram-Schmidt is unstable and is only used here for quick
/// and dirty ortho on matrices that are already almost unitary
pub fn gram_schmidt_ortho<S>(v: &mut DMatrix<S>)
where S: ComplexField{
    let sh = v.shape();
    if sh.0 != sh.1{
        panic!("gram_schimdt_ortho: matrix must be square")
    }
    let n = sh.0;
    //let mut norms : DVector<S::RealField> = DVector::zeros(n);

    v.column_mut(0).normalize_mut();
    for i in 1..n{
        for j in 0..i{
            let c: S = v.column(i).dotc(&v.column(j)) ;
            unsafe{
                for k in 0..n{
                    let s: S = c * *v.get_unchecked((j,k));
                    *v.get_unchecked_mut((i,k)) -= s;
                }
            }
        }
        v.column_mut(i).normalize_mut();
    }

}

pub fn qr_ortho<S>(mut v: DMatrix<S>) -> DMatrix<S>
where S: ComplexField+LapackScalar
{
    let sh = v.shape();
    if sh.0 != sh.1{
        panic!("qr_ortho: matrix must be square")
    }
    let n = sh.0 as i32;
    let mut tau :DVector<S> = DVector::zeros(n as usize);
    let info = S::geqrf(Layout::ColumnMajor, n, n, v.as_mut_slice(),
             n, tau.as_mut_slice());
    if info < 0{
        panic!("Illegal argument error - _geqrf returned {}", info);
    } else if info > 0{
        panic!("Unexpected computation error - _geqrf returned {}", info);
    }

    let info = S::ungqr(Layout::ColumnMajor, n, n, n, v.as_mut_slice(), n,
              tau.as_mut_slice());

    if info < 0{
        panic!("Illegal argument error - _orqqr/_ungqr returned {}", info);
    } else if info > 0{
        panic!("Unexpected computation error - _orqqr/_ungqr returned {}", info);
    }

    v
}

/// Handles a set of k orthogonal vectors obtained from diagonalization of a degenerate
/// eigenspace of a hermitian matrix. Assuming the vectors w_rel are known to be "close" to the degenerate
/// subspace, this finds the basis vectors of the subspace that are as close as possible to
/// the basis w_rel via a QR decomposition of the projection of w_rel
///
///
/// w_deg: (n x k) k orthogonal column vectors of dimension n
/// w_rel: (n x k) relative orthogonal vectors to rotate w_deg onte as close as possible
pub fn handle_degeneracies_qr<S>(w_deg: &DMatrix<S>, w_rel: &DMatrix<S>) -> DMatrix<S>
where S: ComplexField+LapackScalar
{
    let v = w_deg.ad_mul(w_rel);  // k x k w_rel onto w_deg overlap
    let q = qr_ortho(v);  // reorthogonalize in w_deg subspace
    let w_rot_deg = w_deg * &q; // undo the q rotation
    w_rot_deg
}

/// Given a basis w_deg that contains degeneracies described by degens and a full rank w_rel that is preferred
/// find a reorthogonalization of the degenerate eigenspaces of w_deg such that the new basis vectors
/// are "as close as possible" to w_rel.
///
/// In adiabatic simulation, such a basis minimizes the variation in the diabatic perturbations
/// of each time step.
pub fn handle_degeneracies_relative<S>(degens: &DegenArray, w_deg: &mut DMatrix<S>, w_rel: &DMatrix<S>)
    where S: ComplexField+LapackScalar
{
    for di in degens{
        let i0 = di[0];
        let k = di.len();

        let wk_deg = w_deg.columns(i0, k).into_owned();
        let wk_rel =w_rel.columns(i0, k).into_owned();
        let wk_rot_deg = handle_degeneracies_qr(& wk_deg, &wk_rel);

        w_deg.columns_mut(i0, k).copy_from(& wk_rot_deg);
    }
}

pub fn handle_degeneracies_relative_vals<R>(vals: &DVector<R>, w_deg: &mut DMatrix<Complex<R>>,
                                            w_rel: &DMatrix<Complex<R>>, rtol: Option<R>)
        where R: RealField, Complex<R>: ComplexField+LapackScalar{
    let degens = degeneracy_detect(vals, rtol);
    handle_degeneracies_relative(&degens, w_deg, w_rel);
}

pub fn degeneracy_detect<R: RealField>(vals: &DVector<R>, rtol: Option<R>) -> DegenArray{

    let n = vals.len();
    let rtol = match rtol{ Some(r) => r, None => R::from_subset(&1.0e-8)};
    let mut degens : Vec<SmallVec<[usize; 4]>> = Vec::new();

    //apologies for the c style index arithmetic
    let mut i = 0;
    'a: while i < n {
        let vi = vals[i];
        let mut di  = SmallVec::new();
        di.push(i);

        let mut j = i+1;
        'b: while j < n {
            let vj = vals[j];
            let eps : R = ComplexField::abs((vi - vj)/vj);
            if eps < rtol{
                di.push(j);
            } else {
                break 'b;
            }

            j+=1;
        }
        if di.len() > 1 {
            degens.push(di);
        }

        i=j;
    }

    degens
}

///Handle degeneracies in the standard basis
pub fn handle_degeneracies<R: RealField>(degens: &DegenArray, vecs: &mut DMatrix<Complex<R>>)
where R: RealField, Complex<R>: ComplexField+LapackScalar
{

    for di in degens{
        let i0 = di[0];
        let k = di.len();

        let w_deg = vecs.slice((i0, i0), (k, k)).into_owned();
        let q = qr_ortho(w_deg.clone());
        let w_rot_deg = vecs.columns(i0, k) * q;
        vecs.columns_mut(i0, k).copy_from(&w_rot_deg);

    }
    // We re-orthonormolize every group of degenerate eigenstates
//    for di in degens{
//        let i0 = di[0];
//        let k = di.len();
//
//        let w = vecs.slice((i0, i0), (k, k));
//        let mut w = w.into_owned();
//        w.adjoint_mut();
//        gram_schmidt_ortho(&mut w);
//
//        let old_vs = vecs.columns(i0, k).into_owned();
//        let mut vs = vecs.columns_mut(i0, k);
//        old_vs.mul_to(&w, &mut vs);
//    }
}

/// Handle the phases of a matrix by naively setting the largest element to be a
/// positive real number.
pub fn handle_phases<R>(vecs: &mut DMatrix<Complex<R>>)
where R: RealField, Complex<R>: ComplexField+LapackScalar
{
    for mut col in vecs.column_iter_mut(){
        let (i,j) = col.icamax_full();

        let vdag = col[(i,j)].conjugate();
        let vdag_arg = vdag.signum();

        col *= vdag_arg;
    }
}

pub fn handle_relative_phases<R:RealField>(vecs: &mut DMatrix<Complex<R>>,
                                           basis: &DMatrix<Complex<R>>,
                                           temp: &mut DMatrix<Complex<R>>) -> usize
{
    //let v0 = vecs.column(0);
    let mut rank_loss = 0;
    basis.ad_mul_to(&*vecs, temp);
    for (v, mut col) in temp.column_iter().zip(vecs.column_iter_mut()){
        let (i,j) = v.icamax_full();
        let adag = v[(i,j)].conjugate();
        let arg = adag.signum();
        if arg.abs() > R::zero() {
            col *= arg
        } else {
            rank_loss += 1;
        }
    }
    rank_loss
    //handle_phases(temp);
    //basis.mul_to(&*temp, vecs);

    //let v0 = vecs.column(0);
}

pub fn handle_degeneracies_vals<R>(vals: &DVector<R>, vecs: &mut DMatrix<Complex<R>>, rtol: Option<R>)
where R: RealField, Complex<R>: ComplexField+LapackScalar
{
    let degens = degeneracy_detect(&vals, rtol);

    handle_degeneracies(&degens, vecs)
}
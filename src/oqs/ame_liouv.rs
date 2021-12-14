use nalgebra::{DVector, DMatrix};
use qrs_core::{ComplexScalar, RealScalar, ComplexField};
use num_complex::Complex;
use lapack_traits::LapackScalar;
use super::bath::Bath;
use crate::util::{outer_zip_to, copy_transmute_to, change_basis_to};
use qrs_core::quantum::QObj;
use num_traits::Zero;

pub trait Scalar : ComplexScalar + LapackScalar { }
impl<N> Scalar for N where N: ComplexScalar + LapackScalar{ }

#[derive(Copy, Clone)]
pub enum AMEEvalType{
    /// Perform the full evaluation of the AME Liouvillian
    Default,
    /// Evaluate only the rates and Lamb shift
    Simple
}

pub struct AMEWorkpad<R: RealScalar>{
    pub omega: DMatrix<R>,
    pub gamma: DMatrix<R>,
    pub linds_ab: Vec<DMatrix<Complex<R>>>,
    pub gamma_out: DVector<Complex<R>>,
    pub lind_coh_1: DMatrix<Complex<R>>,
    pub lind_coh_2: DMatrix<Complex<R>>,
    pub diag_gamma_0:  DMatrix<Complex<R>>,
    pub lamb_shift: DMatrix<Complex<R>>,
    pub ztemp0: DMatrix<Complex<R>>
}

impl<R: RealScalar> AMEWorkpad<R>{

    pub fn new(n: usize, k: usize) -> Self{
        let mut linds_ab =  Vec::new();
        linds_ab.resize_with(k, || DMatrix::zeros(n,n));

        Self{   omega: DMatrix::zeros(n,n),
            gamma: DMatrix::zeros(n,n),
            linds_ab,
            gamma_out: DVector::zeros(n),
            lind_coh_1: DMatrix::zeros(n,n),
            lind_coh_2: DMatrix::zeros(n,n),
            diag_gamma_0:  DMatrix::zeros(n, n),
            lamb_shift: DMatrix::zeros(n, n),
            ztemp0: DMatrix::zeros(n,n)
        }
    }
}

///         Evaluates the Liouvillian matrices of the Adiabatic Master Equation
///         given the eigenvalues and eigenvectors of the Hamiltonian
///         and coupling operators subject to identical and independent
///         spectral densitise
///  The Liouvillian consists of the Hamiltonian, the Coherent dissipator,
///  and the Incoherent dissipator. In the adiabatic frame, the AME evolution is
///     d(rho_ad)/dt =  (H .* coh(rho_ad) + S .* coh(rho_ad))  (+)  diag( P @ diag(rho_ad))
/// where   H is the Hamiltionian in its own energy basis, i.e. -i omega[a,b]
///         S is the coherent dissipator matrix
///         P is the incoherent dissipator matrix, i.e. the Pauli ME matrix
///
///     H and S have fully coherent action and act via
///         componentwise multiplication .* on the density matrix
///         ** in the instantaneous energy frame only **
///     P is an action only on the populations, i.e. the diagonal of rho, in the instantaneous
///         energy frame.
///
///     This does not include the diabatic contribution, which must be computed by
///     finite difference on the eigenvectors
pub fn ame_liouvillian<R: RealScalar, B: Bath<R>>(
    bath: & B,
    work: &mut AMEWorkpad<R>,
    vals: &DVector<R>,
    vecs: &DMatrix<Complex<R>>,
    haml: &mut DMatrix<Complex<R>>,
    lind_pauli: &mut DMatrix<Complex<R>>,
    lind_coh: &mut DMatrix<Complex<R>>,
    lindblad_ops: &Vec<DMatrix<Complex<R>>>,
    eval_type: AMEEvalType
)
    where Complex<R> : Scalar<R=R>
{
    assert_eq!(lindblad_ops.len(), work.linds_ab.len(), "Number of lindblad operators mismatched");
    let one_half = R::from_subset(&(0.5_f64));

    //                  TRANSITION FREQUENCIES AND HAMILTONIAN
    // In the adiabatic basis -i [H, rho][a,b] = -i omega[a,b] rho[a,b]
    // omega[i,j] = vals[i] - vals[j]
    outer_zip_to(vals, vals, &mut work.omega, |a, b| *a - *b);
    copy_transmute_to(& work.omega, haml);
    //H[a,b] = - i omega[a,b]
    haml.qscal(-Complex::i());
    //DenseQRep::qscal(-Complex::i(), haml);

    //               If no Lindblad Operators, we are done
    if lindblad_ops.len() == 0{
        return;
    }

    //                  LINDBLAD OPERATORS BASIS CHANGE
    //                 Into instantaneous energy basis
    for (l_ab, l0) in work.linds_ab.iter_mut()
        .zip(lindblad_ops.iter()) {
        change_basis_to(l0, vecs, &mut work.ztemp0, l_ab);
    }

    //                  EVALUATE TRANSITION RATES
    //                  g[a, b] = \gamma( w[b, a] )
    for (g, w) in work.gamma.iter_mut().zip(work.omega.iter()){
        *g = bath.gamma(-*w);
    }

    //                  LINDBLAD PAULI MATRIX EVALUATION
    //              pauli_mat = \tilde{\Gamma} - diag(Out_rate)
    //Where
    //          Out_rate[a,b]   =   sum_a g[a,b] Asq[a,b] = sum_a \tilde{Gamma}[a,b]
    //      \tilde{\Gamma}[a,b] =   sum_{alpha} gamma[a,b] A_{alpha}[a, b] A_{alpha}^* [a, b]
    //
    // For an independent and identical baths
    //      \tilde{\Gamma} = sum_{alpha} gamma[a,b] A_{alpha}[a, b] A_{alpha}^* [a, b]
    // i.e.  gamma[a,b]  .*  sum_{alpha} abs( A[a,b] ).^2 where a != b
    //          and is identically 0 along its diagonal
    lind_pauli.apply(|x| *x=Complex::zero());
    for l_ab in work.linds_ab.iter(){
        lind_pauli.zip_apply(l_ab,
                             |s, l| *s += Complex::from(l.norm_sqr()) )
    }

    //
    //  LAMB SHIFT EVALUATION
    //  While we're here:
    //    The quantity sum_{alpha} abs( A[a,b] ).^2 is currently stored in lind_pauli
    //    and can be use to evaluate the lamb_shift
    //
    if bath.has_lamb_shift(){
        for(s, &w) in work.lamb_shift.iter_mut().zip(work.omega.iter()){
            *s = Complex::from(bath.lamb_shift(-w).unwrap());
        }
        work.lamb_shift.component_mul_assign(&lind_pauli);
        let hls = work.lamb_shift.row_sum_tr();

        outer_zip_to(&hls, &hls, &mut work.lamb_shift,
                     |&a, &b| a - b);
        work.lamb_shift *= -Complex::i();
        *haml += &work.lamb_shift;
    }

    // Back to the Pauli rates
    lind_pauli.zip_apply(&work.gamma,
                         |s, g| *s *= Complex::from(g));

    // ** Control Flow Exception **
    // If only simple rates and the lamb shift were requested, we can return now
    // The lind_pauli matrix now has stored the rate information
    //     sum_{alpha} gamma[a,b] A_{alpha}[a, b] A_{alpha}^* [a, b]
    if let AMEEvalType::Simple = eval_type{
        return
    }

    lind_pauli.fill_diagonal(Complex::zero());

    //Out_gamma =sum_a g_ab Asq[a,b] = sum_a tilde{Gamma}[a,b]
    for (g1, g0) in lind_pauli.column_iter()
        .zip(work.gamma_out.iter_mut()){
        *g0 = g1.sum();
    }

    // and finally
    //                  pauli_mat = E_Gamma_offdiag - diag(Out_rate)
    //              *******************************************************
    for i in 0..lind_pauli.nrows(){
        lind_pauli[(i,i)] =  -work.gamma_out[i];
    }


    //                      LINDBLAD COHERENT MATRIX EVALUATION
    //               *******************************************************


    //          Lind_coh_1 = -(1.0 / 2.0) * (Out_rate[:, np.newaxis] + Out_rate[np.newaxis, :])
    //                *******************************************************
    outer_zip_to(&work.gamma_out, &work.gamma_out, &mut work.lind_coh_1,
                 |a, b|
                     - Complex::from(one_half) * ( a + b));

    // diag_A_sum = np.sum(np.conj(diag_A[:,  :, np.newaxis]) * diag_A[:, np.newaxis, :], axis=0)
    for l_ab in work.linds_ab.iter(){
        let n = l_ab.ncols();
        for i in 0..n{
            for j in 0..n{
                work.diag_gamma_0[(i, j)] += l_ab[(i, i)].conjugate() * l_ab[(j, j)];
            }
        }
    }
    //For independent and identical baths
    //G0[a,b] = gamma[0,0] * diag_A_sum
    work.diag_gamma_0.scale_mut(work.gamma[(0, 0)]);

    //
    //                          Lind_coh_2 = G0[a, b]* - (1/2)(G0[a,a] + G0[b,b])
    //                      *******************************************************
    let n = work.lind_coh_2.ncols();
    for i in 0..n{
        for j in 0..n{

            work.lind_coh_2[(i,j)] = work.diag_gamma_0[(j,i)]
                - Complex::from(one_half) * (work.diag_gamma_0[(i,i)] + work.diag_gamma_0[(j,j)] )
        }
    }

    lind_coh.copy_from(&work.lind_coh_1);
    *lind_coh += &work.lind_coh_2;
    lind_coh.fill_diagonal(Complex::zero());
}
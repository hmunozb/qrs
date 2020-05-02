use crate::util::*;
use qrs_core::reps::matrix::*;
use qrs_core::quantum::{QRep, QObj};
use crate::oqs::bath::Bath;
//use crate::ode::dense::DenseExpiSplit;
use crate::ode::super_op::{KineticExpSplit, CoherentExpSplit, DenMatExpiSplit};
use crate::ode::super_op::{MaybeScalePowExp};
use qrs_core::{ComplexField, RealScalar};
use lapack_traits::LapackScalar;
use log::{info, error, warn, trace};
use num_traits::{Zero, Float};
use num_complex::Complex;
use num_complex::Complex64 as c64;
use nalgebra::{DVector, DMatrix};
use smallvec::SmallVec;
use vec_ode::exp::split_exp::{SemiComplexO4ExpSplit, //StrangSplit,
                              CommutativeExpSplit, //TripleJumpExpSplit,
                              RKNR4ExpSplit};
use vec_ode::exp::{DirectSumL, ExponentialSplit};
use ndarray::{ ArrayView2};
use vec_ode::{ODEState, ODEStep, ODESolver, ODESolverBase, ODEError, LinearCombination, LinearCombinationSpace};
use vec_ode::exp::cfm::ExpCFMSolver;
use vec_ode::AdaptiveODESolver;
use crate::util::degen::{handle_degeneracies, degeneracy_detect,
                         handle_relative_phases, handle_degeneracies_relative,
                         handle_degeneracies_relative_vals};
use crate::util::diff::four_point_gl;
use crate::ComplexScalar;
//use alga::linear::NormedSpace;

static AME_DIS_KINETIC_SCALE_LIM : f64 = 5.0e-1;
static AME_HAML_COHERENT_SCALE_LIM : f64 = 1.0e9;

type AMEDissipatorSplit<T> = CommutativeExpSplit<T, Complex<T>, Op<Complex<T>>,
                                MaybeScalePowExp<T, KineticExpSplit<T>>,
                                CoherentExpSplit>;
type AMEHamiltonianSplit<T> = RKNR4ExpSplit<T, Complex<T>, Op<Complex<T>>,
                                CoherentExpSplit,
                                DenMatExpiSplit<T>>;
type AdiabaticMEExpSplit<T> = SemiComplexO4ExpSplit<T, Complex<T>, Op<Complex<T>>,
                                AMEHamiltonianSplit<T>,
                                AMEDissipatorSplit<T>>;

type AdiabaticMEL<T> = <AdiabaticMEExpSplit<T> as ExponentialSplit<T, Complex<T>, Op<Complex<T>>>>::L;

fn make_ame_split<T: RealScalar + Float>(n: u32) -> AdiabaticMEExpSplit<T>
where Complex<T> : ComplexScalar<R=T>
{
    let split = AdiabaticMEExpSplit::<T>::new(
        AMEHamiltonianSplit::new(
            CoherentExpSplit::new(n),
                    DenMatExpiSplit::new(n),
            //DenMatExpiSplit::new(n)
             ),
        AMEDissipatorSplit::new(
             MaybeScalePowExp::new(KineticExpSplit::new(n),
                                   T::from_subset(&AME_DIS_KINETIC_SCALE_LIM)),
             CoherentExpSplit::new(n),)
    );

    split
}

struct AMEWorkpad<R: RealScalar>{
    omega: DMatrix<R>,
    gamma: DMatrix<R>,
    linds_ab: Vec<DMatrix<Complex<R>>>,
    gamma_out: DVector<Complex<R>>,
    lind_coh_1: DMatrix<Complex<R>>,
    lind_coh_2: DMatrix<Complex<R>>,
    diag_gamma_0:  DMatrix<Complex<R>>,
    lamb_shift: DMatrix<Complex<R>>,
    ztemp0: DMatrix<Complex<R>>
}

impl<R: RealScalar> AMEWorkpad<R>{

    fn new(n: usize, k: usize) -> Self{
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
fn ame_liouvillian<R: RealScalar, B: Bath<R>>(
    bath: & B,
    work: &mut AMEWorkpad<R>,
    vals: &DVector<R>,
    vecs: &DMatrix<Complex<R>>,
    haml: &mut DMatrix<Complex<R>>,
    lind_pauli: &mut DMatrix<Complex<R>>,
    lind_coh: &mut DMatrix<Complex<R>>,
    lindblad_ops: &Vec<DMatrix<Complex<R>>>
)
where Complex<R> : ComplexScalar<R=R>
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
    lind_pauli.apply(|_| Complex::zero());
    for l_ab in work.linds_ab.iter(){
        lind_pauli.zip_apply(l_ab,
                                   |s, l| s + Complex::from(l.norm_sqr()) )
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
                               |s, g| s*Complex::from(g));

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



fn assert_orthogonal<T>(v:& DMatrix<Complex<T>>)
where T: RealScalar
{
    let vad = v.adjoint();
    let vv : DMatrix<Complex<T>> = vad * v;

    assert!(vv.is_identity(T::from_subset(&1.0e-9)), "Orthogonal assertion failed");
}

pub struct AMEResults{
    pub t: Vec<f64>,
    pub rho: Vec<Op<c64>>,
    pub partitions: Vec<u32>,
    pub eigvecs: Vec<Op<c64>>,
    pub observables: Vec<Vec<c64>>
}

impl AMEResults{
    pub fn change_rho_basis(&self, new_bases: Vec<Op<c64>>) -> Vec<Op<c64>>{
        let mut new_rhos = Vec::new();
        for ((rho, &p), basis) in self.rho.iter().zip(self.partitions.iter())
                                    .zip(self.eigvecs.iter()){
            let w = &new_bases[p as usize] * basis;
            let new_rho = unchange_basis(rho, &w);
            new_rhos.push(new_rho);
        }
        new_rhos
    }
}

pub struct AME<'a, B: Bath<f64>>{
    haml: &'a TimePartHaml<'a, f64>,
    lindblad_ops: &'a Vec<Op<c64>>,
    p_lindblad_ops: Vec<Op<c64>>,
    work: AMEWorkpad<f64>,
    adiab_haml: Op<c64>,
    diab_k: Op<c64>,
    lind_pauli: Op<c64>,
    lind_coh: Op<c64>,
    eigvecs: Op<c64>,
    eigvals: DVector<f64>,
    bath: &'a B,
    prev_t: Option<(f64,f64)>,
    prev_eigvecs: (Op<c64>, Op<c64>)
}

impl<'a, B: Bath<f64>> AME<'a, B> {
//    pub fn from_operators(haml_mat: TimeDepMatrix<'a, Complex<f64>>,
//                          lindblad_ops: &'a Vec<Op<c64>>){
//        let haml = TimePartHaml::new(TimeDepMatrix,)
//    }

    pub fn new(haml: &'a TimePartHaml<'a, f64>,
               lindblad_ops: &'a Vec<Op<c64>>,
               bath: &'a B) -> Self{
        let n = haml.basis_size() as usize;
        let k = lindblad_ops.len();

        let mut me = Self{haml, lindblad_ops, p_lindblad_ops: Vec::new(),
            work: AMEWorkpad::new(n as usize, k),
            adiab_haml: Op::zeros(n, n),
            diab_k: Op::zeros(n,n),
            lind_pauli: Op::zeros(n, n),
            lind_coh: Op::zeros(n, n),
            eigvecs: Op::zeros(n, n),
            eigvals: DVector::zeros(n),
            bath,
            prev_t: None,
            prev_eigvecs: (Op::zeros(n,n), Op::zeros(n,n))
        };
        me.load_partition(0);
        me
    }

    /// Replace the lindblad operators with the truncated basis of the
    /// pth partition
    /// (todo: should be modified when adaptive basis sizes are available)
    fn load_partition(&mut self, p: usize){
        self.p_lindblad_ops.clear();
        match self.prev_t{
            None => {},
            Some((t0, tf)) =>{
                if p > 0 {
                    //let ep0 = self.haml.advance_partition(&self.prev_eigvecs.0, p-1);
                    //let epf = self.haml.advance_partition(&self.prev_eigvecs.1, p-1);
                    let proj = self.haml.projector(p-1);
                    let ep0 = proj.ad_mul(&self.prev_eigvecs.0);
                    let epf = proj.ad_mul(&self.prev_eigvecs.1);

                    //let eq0 = qr_ortho(ep0);
                    //let eqf = qr_ortho(epf);
                    let (vals0, mut ev0) = self.haml.eig_p(t0, Some(p));
                    let (valsf, mut evf) = self.haml.eig_p(tf, Some(p));
                    handle_degeneracies_relative_vals(&vals0, &mut ev0, &ep0, None);
                    handle_degeneracies_relative_vals(&valsf, &mut evf, &epf, None);
                    let mut rank_loss = handle_relative_phases(&mut ev0, &ep0, &mut self.work.ztemp0);
                    rank_loss += handle_relative_phases(&mut evf, &epf, &mut self.work.ztemp0);
                    if rank_loss > 0{
                        warn!("{} adiabatic rank(s) scattered going into partition {}", rank_loss, p);
                    }
                    self.prev_eigvecs.0 = ev0;
                    self.prev_eigvecs.1 = evf;
                }
            }
        }
        //self.prev_t = None;
        for lind in self.lindblad_ops{
            self.p_lindblad_ops.push(self.haml.transform_to_partition(lind, p ));
        }
    }

    /// Loads the eigenvalues and eigenvectors of time t
    /// No gauge or degeneracy handling
    fn load_eigv(&mut self, t: f64, p: usize, _degen_handle: bool){
        let (vals, vecs) =
            self.haml.eig_p(t, Some(p ));
        self.eigvecs = vecs;
        self.eigvals = vals;
    }

    /// Loads the eigenvalues and eigenvectors of time t
    /// with degeneracy handling
    fn load_eigv_degen_handle(&mut self, t: f64, p: usize, v0: &Op<c64>){
        let (vals, vecs) =
            self.haml.eig_p(t, Some(p ));
        self.eigvecs = vecs;
        self.eigvals = vals;

        let degens = degeneracy_detect(&self.eigvals, None);
        if degens.len() > 0{
            handle_degeneracies_relative(&degens, &mut self.eigvecs,v0);
            assert_orthogonal(& self.eigvecs);
            trace!("A degeneracy at time {} was handled. ", t)
        }

    }
    pub fn last_eigvecs(&self) ->  &Op<c64>{
        &self.prev_eigvecs.1
    }

    pub fn adiabatic_lind(&mut self){
//        let (vals, vecs) =
//            self.haml.eig_p(t, Some(p ), false);
//        self.eigvecs = vecs;
//        self.eigvals = vals;
        ame_liouvillian(self.bath, &mut self.work, &self.eigvals, &self.eigvecs,
                        &mut self.adiab_haml, &mut self.lind_pauli, &mut self.lind_coh,
                        &self.p_lindblad_ops );
    }

    /// Evaluates the diabatic perturbation  -K_A(t)  to second order with spacing dt
    /// In the adiabatic frame,
    ///  K_A = i W^{dag} (dW/dt)
    pub fn diabatic_driver(&mut self, t: f64, p: usize, dt: f64){
        //let n = DenseQRep::qdim_op(&self.adiab_haml);
        let (_vals1, vecs1) = self.haml.eig_p(t-dt, Some(p));
        let (_vals2, vecs2) = self.haml.eig_p(t+dt, Some(p));

        let mut w1 = self.eigvecs.ad_mul(&vecs1);
        let mut w2 = self.eigvecs.ad_mul(&vecs2);
        // Degeneracies have to be handled carefully for the diabatic perturbation
        let degens = degeneracy_detect(&self.eigvals, None);

        if degens.len() > 0{
//            if degens.len() > 0{
//                warn!("\
//ame (diabatic_driver): A possible degeneracy was handled at time {}. \
//This is normal for some diagonal Hamiltonian operators.", t)
//            }
            //let mut w1 = self.eigvecs.ad_mul(&vecs1);
            handle_degeneracies(&degens, &mut w1);
            //m1 = &self.eigvecs * w1;
            //let mut w2 = self.eigvecs.ad_mul(&vecs2);
            handle_degeneracies(&degens, &mut w2);
            //m2 = &self.eigvecs * w2;
        }
//        else {
//            m1 = vecs1;
//            m2 = vecs2;
//        }
        // Handle the phases so that they are consistent with the current eigenvectors
        //handle_phases(&mut w1);
        //handle_phases(&mut w2);

        //let m1 = &self.eigvecs * w1;
        //let m2 = &self.eigvecs * w2;
//        let m1 = w1;
//        let m2 = w2;

//        let mut w_dot = m2;
//        w_dot -= m1;
//        w_dot *= Complex::from(1.0/(2.0 * dt));

//        let degens = degeneracy_detect(&self.eigvals, None);
//        if degens.len() > 0{
//            warn!(
//"\
//A possible degeneracy was handled at time {}. \
//This is normal for some diagonal Hamiltonian operators.", t)
//        }
//        for di in degens.iter(){
//            for &i in di.iter(){
//
//                let mut col = w_dot.column_mut(i);
//                col *= Complex::from(0.0);
//            }
//        }

        let mut w_dot = w2;
        w_dot -= w1;
        //w_dot *= Complex::from(1.0/(2.0 * dt));
        w_dot.scalar_multiply_to(-Complex::<f64>::i() * Complex::from(1.0/(2.0 * dt)),
                                 &mut self.diab_k);
        //self.eigvecs.ad_mul_to(&w_dot, &mut self.diab_k);
        //self.diab_k *= -Complex::i();

        self.diab_k.adjoint_to(&mut self.work.ztemp0);
        self.diab_k += &self.work.ztemp0;
        self.diab_k /= Complex::from(2.0);

    }

    /// This function currently has a lot of very specialized handling for keeping
    /// track of the eigenvectors at the endpoints (t0, tf) of each integration steps
    /// These endpoints must be used to accurately evaluate the diabatic perturbation K_A
    ///
    pub fn generate_split(&mut self, t_arr: &[f64], (t0, tf): (f64, f64), p: usize)
        -> Vec<AdiabaticMEL<f64>>
    {
        if t_arr.len() != 2 {
            panic!("ame: only two-point quadratures are supported")
        }

        let prev_t = self.prev_t;
        let (v0, vf) =
        match prev_t{
            None =>{
                self.load_eigv(t0, p, false);
                let v0 = self.eigvecs.clone();
                self.load_eigv(tf, p,false);
                let mut vf = self.eigvecs.clone();
                handle_relative_phases(&mut vf, &v0, &mut self.work.ztemp0);
                (v0, vf)
            }
            Some((s0, sf)) =>{
                if (s0 == t0) && (sf == tf) { // Unlikely but simple case of the exact same interval
                    let v0 = self.prev_eigvecs.0.clone();
                    let vf = self.prev_eigvecs.1.clone();
                    (v0, vf)
                } else if s0 == t0 { // Solver rejected tf and is attempting a smaller time step
                    let v0 = self.prev_eigvecs.0.clone();
                    self.load_eigv_degen_handle(tf, p, &v0);
                    let mut vf = self.eigvecs.clone();
                    handle_relative_phases(&mut vf, &v0, &mut self.work.ztemp0);
                    (v0, vf)
                } else if t0 == sf { // Solver accepted the previous step
                    let v0 = self.prev_eigvecs.1.clone();
                    self.load_eigv_degen_handle(tf, p, &v0);
                    let mut vf = self.eigvecs.clone();
                    handle_relative_phases(&mut vf, &v0, &mut self.work.ztemp0);
                    (v0, vf)
                } else if sf == tf {
                    warn!("generate_split: An generally impossible branch was reached. (sf==tf)");
                    let mut vf = self.prev_eigvecs.1.clone();
                    self.load_eigv(t0, p, false);
                    let v0 = self.eigvecs.clone();
                    handle_relative_phases(&mut vf, &v0, &mut self.work.ztemp0);
                    (v0, vf)
                } else {
                    self.load_eigv(t0, p, false);
                    let v0 = self.eigvecs.clone();
                    self.load_eigv_degen_handle(tf, p, &v0);
                    let mut vf = self.eigvecs.clone();
                    handle_relative_phases(&mut vf, &v0, &mut self.work.ztemp0);
                    (v0, vf)
                }
            }
        };
//        for (mut v0i, mut vfi) in v0.column_iter_mut()
//                .zip(vf.column_iter_mut()){
//            v0i.normalize_mut();
//            vfi.normalize_mut();
//        }

//        let v00 = v0.column(0);
//        let vf0 = vf.column(0);
//        assert_relative_eq!(v00.norm(), 1.0);
//        assert_relative_eq!(vf0.norm(), 1.0);

        let delta_t = tf - t0;
        let mut eigvecs : SmallVec<[Op<c64>; 4]> = SmallVec::new();
        let mut dsl_vec = Vec::new();
        let mut haml_vec = Vec::new();
        let mut l_vec: Vec<AdiabaticMEL<f64>> = Vec::new();

        // Load the eigenvector for the inner quadrature and evaluate
        // the dissipative operators and the adiabatic hamiltonian
        eigvecs.push(v0.clone());
        for &t in t_arr.iter(){
            self.load_eigv_degen_handle(t, p, &v0);
            handle_relative_phases(&mut self.eigvecs, &v0, &mut self.work.ztemp0);

            self.adiabatic_lind();
            //self.lind_pauli.row_sum().map(|s| { assert_relative_eq!(s.abs(), 0.0)});

            let dsl_lind = DirectSumL::new(self.lind_pauli.clone(), self.lind_coh.clone());
            dsl_vec.push(dsl_lind);
            haml_vec.push(self.adiab_haml.clone());
            eigvecs.push(self.eigvecs.clone());
        }
        eigvecs.push(vf.clone());

        let mut dvecs0 = v0.clone();
        let mut dvecs1 = v0.clone();

        // Evaluate the derivatives of the eigenvectors numerically at
        // the assumed two-point Gauss-Legendre quadrature points
        // and additionally using the endpoints t0, tf
        four_point_gl(eigvecs.as_slice(), &mut dvecs0, &mut dvecs1);
        dvecs0 /= Complex::from(delta_t);
        dvecs1 /= Complex::from(delta_t);
        // Finally, evaluate the diabatic perturbation -K_A = -i W^{dag} dW/dt
        // at the quadrature points
        let dvecs = [dvecs0, dvecs1];
        for (i, (dv,(haml, dsl_lind))) in dvecs.iter()
                .zip(haml_vec.into_iter().zip(dsl_vec.into_iter()))
                .enumerate(){
            ad_mul_to(&eigvecs[i+1], &dv, &mut self.work.ztemp0);
            //eigvecs[i+1].ad_mul_to(dv, &mut self.work.ztemp0);
            self.work.ztemp0 *= -Complex::i();
            let mut ka = self.work.ztemp0.adjoint();
            ka += &self.work.ztemp0;
            ka *= Complex::from(1.0/2.0);
            let dsl_haml = DirectSumL::new(haml, ka);
            let dsl = DirectSumL::new(dsl_haml, dsl_lind);
            l_vec.push(dsl);
        }

//        let dt_vec : Vec<f64> = t_arr.iter().zip(t_arr.iter().skip(1))
//            .map(|(a, b)| *b - *a).collect_vec();
//        let dt_arr = Array1::from(dt_vec);
//        let dt_mean = (dt_arr).mean().unwrap();


//        for (i, &t) in t_arr.iter().enumerate(){
//            self.load_eigv(t, p, false);
//
////            if i > 0 { // handle degeneracies and phases to be consistent with the first eigenbasis
////                let degens = degeneracy_detect(&self.eigvals, None);
////                let ref_eigvecs: &Op<c64> = &(l_vec[0].a.b);
////                let mut w = ref_eigvecs.ad_mul(&self.eigvecs);
////                if degens.len() > 0{
////                    warn!("\
////ame: A possible degeneracy was handled at time {}. \
////This is normal for some diagonal Hamiltonian operators.", t);
////                    handle_degeneracies(&degens, &mut w);
////                }
////                handle_phases(&mut w);
////                self.eigvecs = ref_eigvecs * w;
////            }
//
//            self.adiabatic_lind();
//            let dsl_lind = DirectSumL::new(self.lind_pauli.clone(), self.lind_coh.clone());
//            self.diabatic_driver(t, p, dt_mean/2.0);
//            let mut k = self.diab_k.clone();
//            k *= -Complex::i();
//            let dsl_haml = DirectSumL::new(self.adiab_haml.clone(), k);
//            let dsl = DirectSumL::new(dsl_haml, dsl_lind);
//            l_vec.push(dsl);
//        }

        self.prev_eigvecs.0 = v0;
        self.prev_eigvecs.1 = vf;
        self.prev_t = Some((t0,tf));
        l_vec
    }

}

pub fn solve_ame<B: Bath<f64>>(
    ame: &mut AME<B>,
    initial_state: Op<c64>,
    tol: f64,
    dt_max_frac: f64
)
    -> Result<AMEResults, ODEError>
{
    let partitions  = ame.haml.partitions().to_owned();
    let num_partitions = partitions.len() - 1;
    let n = ame.haml.basis_size() as usize;
    let mut norm_est = condest::Normest1::new(n, 2);
    let mut rho0 = initial_state;
    let mut last_delta_t : Option<f64> = None;
    //let mut last_eigs : Op<c64> = Op::<f64>::zeros(n,n);
    let mut rho_vec: Vec<Op<c64>> = Vec::new();
    let mut eigvecs: Vec<Op<c64>> = Vec::new();
    let mut results_parts = Vec::new();


    rho_vec.push(rho0.clone());
    eigvecs.push(ame.haml.eig_p(partitions[0], Some(0)).1);

    for (p, (&ti0, &tif)) in partitions.iter()
            .zip(partitions.iter().skip(1))
            .enumerate()
    {
        let delta_t = tif - ti0;
        let max_step = dt_max_frac * delta_t;
        let min_step = max_step * 1.0e-10;
        let dt = match last_delta_t{
            None => dt_max_frac * delta_t / 20.0,
            Some(dt) => dt.max(1.001*min_step).min(0.999*max_step) };

        info!("Evolving partition {} / {} \t ({}, {})", p+1, num_partitions,
                ti0, tif);

        ame.load_partition(p);
        let split = make_ame_split(n as u32);

        let f = |t_arr: &[f64], (t0, tf) : (f64, f64)| {
                ame.generate_split(t_arr, (t0,tf),p)
            };
        let norm_fn = |m: &Op<c64>| -> f64{
            //should technically be transposed to row major but denmats are hermitian
            //let absm = m.map(|c|c.abs());
                //absm.row_sum().amax()/ ( (n as f64).sqrt())
                //m.lp_norm(1)
                let arr: ArrayView2<_> = ArrayView2::from_shape((n, n), m.as_slice()).unwrap();
                norm_est.normest1(&arr, 5) / ( (n as f64).sqrt())

        };

        let mut rhodag = rho0.clone();
        let mut solver = ExpCFMSolver::new(
            f, norm_fn,ti0, tif, rho0,  dt, split)
            .with_tolerance(1.0e-6, tol)
            .with_step_range(min_step,
                             max_step)
            .with_init_step(dt)
            ;

        //let mut iters = 0;
        //let mut rejects = 0;
        let mut ema_rej :f64 = 0.0;
        loop{
            let res = solver.step();

            match res{
                ODEState::Ok(step) => {
                    match step{
                        ODEStep::Step(dt) => {
                            trace!("t={}\tStepped by {}", solver.ode_data().t, dt);
                            ema_rej = ema_rej * 0.90;
                        },
                        ODEStep::Reject => {
                            ema_rej = ema_rej*0.90 + 0.10*1.0;
                            if ema_rej > 0.9{
                                panic!("Rejection rate too large.");
                            } else if ema_rej > 0.75 {
                                warn!("t={} *** High rejection rate ***\n\t (90% EMA: {})\t\tRejected step size: {}",
                                      solver.ode_data().t, ema_rej, solver.ode_data().next_dt)
                            }
                            else if ema_rej > 0.5 {
                                trace!("t={} *** Elevated rejection rate ***\n\t (90% EMA: {})\t\tRejected step size: {}",
                                      solver.ode_data().t, ema_rej, solver.ode_data().next_dt)
                            }
//                            let rej_rate = rejects as f64 / iters as f64;
//                            if iters > 10 && (rej_rate > 0.4) {
//                                warn!("t={}\tHigh rejection rate ({}). Rejected step size: {}",
//                                      solver.ode_data().t, rej_rate, solver.ode_data().next_dt)
//                            }
                        },
                        _ => {}
                    }
                },
                ODEState::Done => {
                    trace!("Partition Done");
                    break;
                },
                ODEState::Err(e) => {
                    error!("ODE solver error occurred.");
                    return Err(e);
                }
            }
            //Enforce Hermiticity after each step
            let dat = solver.ode_data_mut();
            dat.x.adjoint_to(&mut rhodag);
            dat.x += &rhodag;
            dat.x /= c64::from(2.0);
        }
//
//        while let ODEState::Ok(step) = solver.step()
//
//            {
//            //Enforce Hermiticity after each step
//            let mut dat = solver.ode_data_mut();
//            dat.x.adjoint_to(&mut rhodag);
//            dat.x += &rhodag;
//            dat.x /= c64::from(2.0);
//        }

        last_delta_t = Some(solver.ode_data().h);
        let(_, rhof) = solver.into_current();
        rho_vec.push(rhof.clone());
        eigvecs.push(ame.last_eigvecs().clone());
        results_parts.push(p as u32);
        //if p + 1 < num_partitions {
        //    rho0 = ame.haml.advance_partition(&rhof, p);
        //} else {
        rho0 = rhof;
        //}
    }

    let results = AMEResults{t: partitions, rho: rho_vec,
        partitions: results_parts, eigvecs, observables: Vec::new() };

    Ok(results)
}

#[cfg(test)]
mod tests{
    use num_complex::Complex;
    use super::*;
    use crate::base::quantum::QRep;
    use qrs_core::reps::dense::*;
    use crate::base::pauli::matrix as pauli;
    use crate::oqs::bath::OhmicBath;
    use crate::ode::super_op::DenMatExpiSplit;
    use crate::util::{TimeDepMatrixTerm, TimeDepMatrix, TimePartHaml};
    use crate::oqs::adiabatic_me::{AME, solve_ame, make_ame_split};
    use alga::general::RealField;
    use num_complex::Complex64 as c64;
    use num_traits::real::Real;
    use vec_ode::LinearCombination;

    extern crate simple_logger;

    #[test]
    fn single_qubit_time_ind_ame(){
        let eta = 1.0e-4;
        let omega_c = 2.0 * f64::pi() * 4.0 ;
        let temp_mk = 12.1;

        let temperature = temp_mk * 0.02084 * 2.0 * f64::pi();
        let beta = temperature.recip();
        let bath = OhmicBath::new(eta, omega_c, beta);

        let tf = 10.0;
        let dt = 0.01;
        let mut sp_haml = DenMatExpiSplit::<f64>::new(2);
        let id = pauli::id::<f64>();
        let sx = pauli::sx::<f64>();
        let sy = pauli::sy::<f64>();
        let sz = pauli::sz::<f64>();

        let fx = |t: f64| Complex::from(0.0);
        let fy = |t: f64| Complex::from(2.0 * f64::pi());

        let hx = TimeDepMatrixTerm::new(&sx, &fx);
        let hy = TimeDepMatrixTerm::new(&sy, &fy);
        let lz = sz.clone();
        let lind_ops = vec![lz];


        let haml = TimeDepMatrix::new(vec![hx, hy]);
        let haml_part = TimePartHaml::new(haml, 2, 0.0, tf,1);

        let mut ame = AME::new(&haml_part, &lind_ops, &bath);
        let rho0 = (id.clone() + &sy)/c64::from(2.0);

//        let mut split = make_ame_split(2);
//        let mut spl1 = ame.generate_split(&[0.50*tf, 0.51*tf], (),0);
//        spl1[0].scale(c64::from(0.1));
//
//        let spu1 = split.exp(&spl1[0]);
//
//        let rho0 = (id.clone() + &sy)/c64::from(2.0);
//        let rho1 = split.map_exp(&spu1, &rho0);
//
//        println!("Initial adiabatic density matrix:\n{}", rho0);
//        println!("Action at tf/2:\n{}", rho1);

        let rhof = solve_ame(
            &mut ame, rho0, 1.0e-6, 0.1);

        match rhof{
            Ok(res) => println!("Final density matrix:\n{}", res.rho.last().unwrap()),
            Err(e) => println!("An ODE Solver error occurred")
        }
    }

    #[test]
    fn single_qubit_ame(){
        simple_logger::init_with_level(log::Level::Info).unwrap();

        let eta = 0.0;
        let omega_c = 2.0 * f64::pi() * 4.0 ;
        let temp_mk = 12.1;

        let temperature = temp_mk * 0.02084 * 2.0 * f64::pi();
        let beta = temperature.recip();
        let bath = OhmicBath::new(eta, omega_c, beta)
            .with_lamb_shift(-12.0*omega_c, 12.0*omega_c)
            ;

        let tf = 10.0 / f64::sqrt(2.0);
        let dt = 0.01;
        let mut sp_haml = DenMatExpiSplit::<f64>::new(2);
        let id = pauli::id::<f64>();
        let sx = pauli::sx::<f64>();
        let sy = pauli::sy::<f64>();
        let sz = pauli::sz::<f64>();

        let fz = |t: f64| Complex::from(-2.0 * f64::pi() *(4.0 * f64::pi() * t).cos());
        let fx = |t: f64| Complex::from(-2.0 * f64::pi() *(4.0 * f64::pi() * t).sin());

        let hz = TimeDepMatrixTerm::new(&sz, &fz);
        let hx = TimeDepMatrixTerm::new(&sx, &fx);
        let lz = sz.clone();
        let lind_ops = vec![lz];


        let haml = TimeDepMatrix::new(vec![hz, hx]);
        let haml_part = TimePartHaml::new(haml, 2, 0.0, tf,20);

        let mut ame = AME::new(&haml_part, &lind_ops, &bath);

        let rho0 = (id.clone() + sz.clone())/c64::from(2.0);
        println!("Initial adiabatic density matrix:\n{}", rho0);

//        let mut split = make_ame_split(2);
//        let mut spl1 = ame.generate_split(&[0.50*tf, 0.51*tf], 0);
//        spl1[0].scale(c64::from(0.05));
//
//        let spu1 = split.exp(&spl1[0]);
//
//        let rho1 = split.map_exp(&spu1, &rho0);
//
//        println!("Action at tf/2:\n{}", rho1);

        let rhof = solve_ame(&mut ame, rho0, 1.0e-10, 0.1);
        match rhof{
            Ok(res) => {
                let rho = res.rho.last().unwrap();
                let mut tr = 0.0; let mut pops = Vec::new();
                let n = rho.shape().0;
                for i in 0..n{
                    let &p = rho.get((n+1)*i).unwrap();
                    let p = p.re;
                    tr += p;
                    pops.push(p);
                }
                println!("Final trace: {}\n\n", tr);

                println!("Final density matrix:\n{}", rho)
            },
            Err(e) => println!("An ODE Solver error occurred")
        }
    }
}
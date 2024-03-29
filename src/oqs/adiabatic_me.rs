use lapack_traits::LapackScalar;
use log::{error, info, debug, trace, warn, log_enabled};
use nalgebra::{DMatrix, DVector};
use ndarray::ArrayView2;
use num_complex::Complex;
use num_complex::Complex64 as c64;
use num_traits::{Float, Zero};
use smallvec::SmallVec;
use serde::{Serialize, Deserialize};
use vec_ode::lc::{LinearCombination, LinearCombinationSpace};
use vec_ode::{ODEError, ODESolver, ODEState, ODEStep};
use vec_ode::AdaptiveODESolver;
use vec_ode::exp::{DirectSumL, ExponentialSplit};
use vec_ode::exp::cfm::ExpCFMSolver;
use vec_ode::exp::split_exp::{CommutativeExpSplit, //StrangSplit,
                              RKNR4ExpSplit, //TripleJumpExpSplit,
                              SemiComplexO4ExpSplit};

use qrs_core::{ComplexField, RealScalar};
use qrs_core::quantum::{QObj, QRep, LinearOperator};
use qrs_core::reps::matrix::*;

use crate::ComplexScalar;
//use crate::ode::dense::DenseExpiSplit;
use crate::ode::super_op::{CoherentExpSplit, DenMatExpiSplit, KineticExpSplit};
use crate::ode::super_op::MaybeScalePowExp;
use crate::oqs::bath::Bath;
use crate::oqs::linops::{DiagonalLinOp, SparseLinOp};
use crate::util::*;
use crate::util::degen::{degeneracy_detect, handle_degeneracies,
                         handle_degeneracies_relative, handle_degeneracies_relative_vals,
                         handle_relative_phases};
use crate::util::diff::four_point_gl;
use crate::adiab::{diabatic_driver, AdiabaticPartitioner};

use super::ame_liouv::{Scalar, AMEWorkpad, ame_liouvillian};
use crate::oqs::ame_liouv::AMEEvalType;
//use alga::linear::NormedSpace;

static AME_DIS_KINETIC_SCALE_LIM : f64 = 5.0e-1;
//static AME_HAML_COHERENT_SCALE_LIM : f64 = 1.0e9;

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
where Complex<T> : Scalar<R=T>
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

#[derive(Serialize, Deserialize)]
pub struct AMEResults{
    pub t: Vec<f64>,
    pub rho: Vec<Op<c64>>,
    pub partitions: Vec<u32>,
    pub eigvecs: Vec<Op<c64>>,
    pub tgt_ampls: Vec<Op<c64>>,
    pub observables: Vec<Vec<c64>>
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AMEState{
    pub t: f64,
    pub p: i32,
    pub rho: Op<c64>,
    pub eigvecs: Op<c64>,
    pub tgt_ampls: Option<Op<c64>>,
    //pub observables: Vec<c64>
}


pub enum AMESparseLindOp<N: ComplexScalar>{
    Sparse(SparseLinOp<N>),
    Diag(DiagonalLinOp<N>)
}

impl<N: ComplexScalar> LinearOperator<Ket<N>, N> for AMESparseLindOp<N>{
    fn map(&self, v: &Ket<N>) -> Ket<N> {
        match self{
            Self::Sparse(l)=> l.map(v),
            Self::Diag(l) => l.map(v)
        }
    }

    fn conj_map(&self, v: &Ket<N>) -> Ket<N> {
        match self{
            Self::Sparse(l)=> l.conj_map(v),
            Self::Diag(l) => l.conj_map(v)
        }
    }

    fn positive_map(&self, v: &Ket<N>) -> Ket<N> {
        match self{
            Self::Sparse(l)=> l.positive_map(v),
            Self::Diag(l) => l.positive_map(v)
        }
    }

    fn add_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        match self{
            Self::Sparse(l)=> l.add_map_to(v, target),
            Self::Diag(l) => l.add_map_to(v, target)
        }
    }

    fn add_conj_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        match self{
            Self::Sparse(l)=> l.add_conj_map_to(v, target),
            Self::Diag(l) => l.add_conj_map_to(v, target)
        }
    }

    fn add_positive_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        match self{
            Self::Sparse(l)=> l.add_positive_map_to(v, target),
            Self::Diag(l) => l.add_positive_map_to(v, target)
        }
    }

    fn positive_ev(&self, v: &Ket<N>) -> <N as ComplexScalar>::R {
        match self{
            Self::Sparse(l)=> l.positive_ev(v),
            Self::Diag(l) => l.positive_ev(v)
        }
    }
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
    // haml: &'a TimePartHaml<'a, f64>,
     a_eigvecs: Op<c64>,
     a_eigvals: DVector<f64>,
    // prev_t: Option<(f64,f64)>,
    // prev_eigvecs: (Op<c64>, Op<c64>),

    pub adb: AdiabaticPartitioner<'a>,

    lindblad_ops: &'a Vec<Op<c64>>,
    p_lindblad_ops: Vec<Op<c64>>,
    work: AMEWorkpad<f64>,
    adiab_haml: Op<c64>,
    //diab_k: Op<c64>,
    lind_pauli: Op<c64>,
    lind_coh: Op<c64>,


    bath: &'a B,

}

impl<'a, B: Bath<f64>> AME<'a, B> {

    pub fn new(haml: &'a TimePartHaml<'a, f64>,
               lindblad_ops: &'a Vec<Op<c64>>,
               bath: &'a B) -> Self{
        let n = haml.basis_size() as usize;
        let k = lindblad_ops.len();
        let adb = AdiabaticPartitioner::new(haml);

        let mut me = Self{adb,
            lindblad_ops, p_lindblad_ops: Vec::new(),
            work: AMEWorkpad::new(n as usize, k),
            adiab_haml: Op::zeros(n, n),
            //diab_k: Op::zeros(n,n),
            lind_pauli: Op::zeros(n, n),
            lind_coh: Op::zeros(n, n),
             a_eigvecs: Op::zeros(n, n),
                a_eigvals: DVector::zeros(n),
            bath,
            // prev_t: None,
            // prev_eigvecs: (Op::zeros(n,n), Op::zeros(n,n))
        };
        me.load_partition(0);
        me
    }

    pub fn gibbs_state(&mut self, t: f64, beta: f64){
        let (vals, vecs) = self.adb.eigv(t);

    }

    pub fn workpad(&self) -> &AMEWorkpad<f64>{
        return &self.work;
    }
    /// Replace the lindblad operators with the truncated basis of the
    /// pth partition
    /// (todo: should be modified when adaptive basis sizes are available)
    pub fn load_partition(&mut self, p: usize){
        self.adb.load_partition(p);
        self.p_lindblad_ops.clear();
        for lind in self.lindblad_ops{
            self.p_lindblad_ops.push(
                self.adb.to_current_basis(lind));
        }
    }

    ///// Loads the eigenvalues and eigenvectors of time t
    ///// with degeneracy handling
    // fn load_eigv_degen_handle(&mut self, t: f64, _p: usize, v0: &Op<c64>){
    //     let (vals, vecs) = self.adb.eigv_degen_handle(t, v0);
    //     self.a_eigvals = vals;
    //     self.a_eigvecs = vecs;
    // }
    fn load_eigv_normal(&mut self, t: f64){
        let (vals, vecs) = self.adb.normalized_eigv(t);
        self.a_eigvals = vals;
        self.a_eigvecs = vecs;
    }

    pub fn last_eigvecs(&self) ->  &Op<c64>{
        &self.adb.interval_eigvecs.1
    }

    pub fn adiabatic_lind(&mut self){
        ame_liouvillian(self.bath, &mut self.work, &self.a_eigvals, &self.a_eigvecs,
                        &mut self.adiab_haml, &mut self.lind_pauli, &mut self.lind_coh,
                        &self.p_lindblad_ops, AMEEvalType::Default
        );
    }

    // /// Evaluates the diabatic perturbation  -K_A(t)  to second order with spacing dt
    // /// In the adiabatic frame,
    // ///  K_A = i W^{dag} (dW/dt)
    // pub fn diabatic_driver(&mut self, t: f64, p: usize, dt: f64){
    //     diabatic_driver(t, p, dt, &mut self.haml, & self.eigvals, &self.eigvecs, &mut self.diab_k);
    // }

    pub fn generate_sparse_lindops(&mut self, t: f64, dt: f64) -> (Op<c64>, Vec<AMESparseLindOp<c64>>){
        // Use the normalized eigvecs calculated for the diabatic driver;
        let n = self.adiab_haml.shape().0;
        let mut ka: Op<c64> = Op::zeros(n, n);
        let mut haml: Op<c64> = Op::zeros(n, n);
        self.adb.diabatic_driver(t, dt, &mut ka);
        let vals = &self.adb.eigvals;
        let vecs = &self.adb.eigvecs;
        ka *= -c64::i();
        self.a_eigvals = vals.clone();
        self.a_eigvecs = (*vecs).clone();
        for i in 0..n{
            unsafe{ *haml.get_unchecked_mut((i, i)) = (*self.a_eigvals.get_unchecked_mut(i)).into()};
        }
        haml *= -c64::i();
        haml += ka;
        // \sum_{alpha} g[a,b] A[a,b]
        ame_liouvillian(self.bath, &mut self.work, &self.a_eigvals, &self.a_eigvecs,
                        &mut self.adiab_haml, &mut self.lind_pauli, &mut self.lind_coh,
                        &self.p_lindblad_ops, AMEEvalType::Simple);

        let mut lind_ops = Vec::with_capacity(n*(n-1) + 1);
        let mut lind_diag: Ket<c64> = Ket::zeros(n);
        for (i,d) in lind_diag.iter_mut().enumerate(){
            *d = self.lind_pauli[(i, i)];
        }
        lind_ops.push(AMESparseLindOp::Diag(DiagonalLinOp{ diag: lind_diag }));
        for i in 0..n{
            for j in 0..n{
                if i != j {
                    lind_ops.push(AMESparseLindOp::Sparse(
                        SparseLinOp{i: i as u32, j: j as u32, g: self.lind_pauli[(i, j)]}))
                }
            }
        }

        return (haml, lind_ops);

    }

    /// This function currently has a lot of very specialized handling for keeping
    /// track of the eigenvectors at the endpoints (t0, tf) of each integration steps
    /// These endpoints must be used to accurately evaluate the diabatic perturbation K_A
    ///
    pub fn generate_split(&mut self, t_arr: &[f64], (t0, tf): (f64, f64))
        -> Vec<AdiabaticMEL<f64>>
    {
        if t_arr.len() != 2 {
            panic!("ame: only two-point quadratures are supported")
        }

        self.adb.step_eigvs(t0, tf);
        let (v0, vf) = self.adb.eigv_interval();
        let v0 = v0.clone_owned();
        let vf = vf.clone_owned();

        let delta_t = tf - t0;
        let mut eigvecs : SmallVec<[Op<c64>; 4]> = SmallVec::new();
        let mut dsl_vec = Vec::new();
        let mut haml_vec = Vec::new();
        let mut l_vec: Vec<AdiabaticMEL<f64>> = Vec::new();
        let mut ztemp0 = v0.clone();

        // Load the eigenvector for the inner quadrature and evaluate
        // the dissipative operators and the adiabatic hamiltonian
        eigvecs.push(v0.clone());
        for &t in t_arr.iter(){
            self.load_eigv_normal(t);
            self.adiabatic_lind();
            if log_enabled!(log::Level::Debug) {
                debug!("tr pauli: {}", self.lind_pauli.trace() );
            }

            let dsl_lind = DirectSumL::new(self.lind_pauli.clone(), self.lind_coh.clone());
            dsl_vec.push(dsl_lind);
            haml_vec.push(self.adiab_haml.clone());
            eigvecs.push(self.a_eigvecs.clone());
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
            ad_mul_to(&eigvecs[i+1], &dv, &mut ztemp0);
            //eigvecs[i+1].ad_mul_to(dv, &mut self.work.ztemp0);
            ztemp0 *= -Complex::i();
            let mut ka = ztemp0.adjoint();
            ka += &ztemp0;
            ka *= Complex::from(1.0/2.0);
            let dsl_haml = DirectSumL::new(haml, ka);
            let dsl = DirectSumL::new(dsl_haml, dsl_lind);
            l_vec.push(dsl);
        }

        l_vec
    }

}

// todo: move this somewhere else
pub fn basis_tgts_ampls(basis_tgts: Option<&Vec<usize>>, rho: &Op<c64>, eigv: &Op<c64>, haml: &TimePartHaml<f64>, p: usize) -> Option<Op<c64>>{
    if let Some(basis_tgts_vec) = basis_tgts{
        return if basis_tgts_vec.len() > 0 {
            let tgt_proj = haml.sparse_canonical_basis_amplitudes(basis_tgts_vec, p);
            let eigvn = tgt_proj * eigv;
            Some(unchange_basis(rho, &eigvn))
        } else {
            None
        }
    }

    None
}

fn inner_solver_loop<Solver>(solver: &mut Solver) -> Result<(),ODEError>
where Solver: AdaptiveODESolver<f64, RangeType=Op<c64>>
{
    let mut rhodag : Op<c64> = solver.ode_data().x.clone();
    let mut ema_rej = 0.0;
    loop{
        let res = solver.step_adaptive();
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
                    },
                    _ => {}
                }
            },
            ODEState::Done => {
                debug!("Partition Done");
                break;
            },
            ODEState::Err(e) => {
                error!("ODE solver error occurred.");
                return Err(e);
            }
        }
        let dat = solver.ode_data_mut();
        dat.x.adjoint_to(&mut rhodag);
        dat.x += &rhodag;
        dat.x /= c64::from(2.0);
    }

    Ok(())
}

/// Solves the adiabatic master equation constructed according to the provided
/// partitioned Hamiltonian, Lindblad operators, and bath spectral density
pub fn solve_ame<B: Bath<f64>>(
    ame: &mut AME<B>,
    initial_state: Op<c64>,
    tol: f64,
    dt_max_frac: f64,
    basis_tgts: Option<&Vec<usize>>,
    mut callback: Option<&mut dyn FnMut(&AMEState)>
)
    -> Result<Vec<AMEState>, ODEError>
{
    let partitions  = ame.adb.haml.partitions().to_owned();
    let num_partitions = partitions.len() - 1;
    let n = ame.adb.haml.basis_size() as usize;
    let t0 = partitions[0];
    let mut norm_est = condest::Normest1::new(n, 2);
    let mut rho0 = initial_state;
    let mut last_delta_t : Option<f64> = None;
    let mut results_vec  = Vec::new();

    // n x n
    let eigv = ame.adb.haml.eig_p(t0, Some(0)).1;
    // tgts x n
    let tgt_ampls = basis_tgts_ampls(basis_tgts, &rho0, &eigv, &ame.adb.haml, 0);

    let ame_state = AMEState{
        t: t0,
        p: 0,
        rho: rho0.clone(),
        eigvecs: eigv.clone(),
        tgt_ampls,
    };
    callback.as_mut().map(|f| f(&ame_state));
    results_vec.push(ame_state);

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

        let norm_fn = |m: &Op<c64>| -> f64{
            //should technically be transposed to row major but denmats are hermitian
            let arr: ArrayView2<_> = ArrayView2::from_shape((n, n), m.as_slice()).unwrap();
            norm_est.normest1(&arr, 5) / ( (n as f64).sqrt())
        };

        let rhodag = rho0.clone();
        let mut solver = ExpCFMSolver::new(
            |t_arr: &[f64], (t0, tf) : (f64, f64)| {
                ame.generate_split(t_arr, (t0,tf))
            },
            norm_fn,ti0, tif, rho0,  dt, split)
            .with_tolerance(1.0e-6, tol)
            .with_step_range(min_step,
                             max_step)
            .with_init_step(dt)
            ;
        inner_solver_loop(&mut solver)?;

        last_delta_t = Some(solver.ode_data().h);
        // End of solver lifetime and ame borrow
        let(_, rhof) = solver.into_current();

        let last_eigvecs = ame.last_eigvecs();
        let tgt_ampls = basis_tgts_ampls(basis_tgts, &rhof,last_eigvecs, &ame.adb.haml, p);

        let ame_state = AMEState{
            t: tif,
            p: p as i32,
            rho: rhof.clone(),
            eigvecs: (*last_eigvecs).clone(),
            tgt_ampls,
        };

        callback.as_mut().map(|f| f(&ame_state));
        results_vec.push(ame_state);
        rho0 = rhof;
    }

    Ok(results_vec)
}

#[cfg(test)]
mod tests{
    use alga::general::RealField;
    use num_complex::Complex;
    use num_complex::Complex64 as c64;
    use num_traits::real::Real;
    use vec_ode::LinearCombination;

    use qrs_core::reps::dense::*;

    use crate::base::pauli::matrix as pauli;
    use crate::base::quantum::QRep;
    use crate::ode::super_op::DenMatExpiSplit;
    use crate::oqs::adiabatic_me::{AME, make_ame_split, solve_ame};
    use crate::oqs::bath::OhmicBath;
    use crate::util::{TimeDepMatrix, TimeDepMatrixTerm, TimePartHaml};

    use super::*;

    use simple_logger::SimpleLogger;

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
        let sp_haml = DenMatExpiSplit::<f64>::new(2);
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

        let rhof = solve_ame(
            &mut ame, rho0, 1.0e-6, 0.1, None, None);

        match rhof{
            Ok(res) => println!("Final density matrix:\n{}", res.last().unwrap().rho),
            Err(e) => println!("An ODE Solver error occurred")
        }
    }

    #[test]
    fn single_qubit_ame(){
        let eta = 1.0e-4;
        let omega_c = 2.0 * f64::pi() * 4.0 ;
        let temp_mk = 12.1;

        let temperature = temp_mk * 0.02084 * 2.0 * f64::pi();
        let beta = temperature.recip();
        let bath = OhmicBath::new(eta, omega_c, beta)
            .with_lamb_shift(-12.0*omega_c, 12.0*omega_c)
            ;

        let tf = 10.0 / f64::sqrt(2.0);
        let dt = 0.01;
        let sp_haml = DenMatExpiSplit::<f64>::new(2);
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

        let rhof = solve_ame(&mut ame, rho0, 1.0e-6, 0.1, None, None);
        match rhof{
            Ok(res) => {
                let rho = &res.last().unwrap().rho;
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
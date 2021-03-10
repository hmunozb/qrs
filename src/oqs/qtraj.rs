use crate::{RealScalar, ComplexScalar};
use crate::base::quantum::{QRep, QObj, QRSLinearCombination, QKet, NormedQRep};
use qrs_core::quantum::LinearOperator;
use vec_ode::exp::{MidpointExpLinearSolver, ExponentialSplit};
use vec_ode::{RK45Solver, ODESolver, ODEState, ODESolverBase, AdaptiveODESolver, ODEData, ODEStep};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::Uniform;
use num_traits::{Zero, One, NumCast};
use itertools::Itertools;
use std::marker::PhantomData;
use num_traits::real::Real;

pub enum JumpType<T: Copy, I>{
    NoJump(T),
    Jump(T,I)
}
impl<T: Copy, I> JumpType<T, I>{
    pub fn t(&self) -> T{
        match self{
            Self::NoJump(t) => *t,
            Self::Jump(t, _) => *t
        }
    }
}

pub struct QtrajPoissonUnravel<R: RealScalar, C: ComplexScalar,
        Q: NormedQRep<C>, L: LinearOperator<Q::KetRep, C>,
    F: FnMut(R)->(Q::OpRep, Vec<L>)>
    where C::RealField : SampleUniform
{
    lind_op_fn: F,
    rand_uniform: Uniform<C::RealField>,
    rtol: C::R,
    atol: C::R,
    last_lind_ops: Option<Vec<L>>,
    _phantom: PhantomData<(Q, R)>
}

impl < R: RealScalar, C: ComplexScalar, Q: NormedQRep<C>,
    L: LinearOperator<Q::KetRep, C>,
    F: FnMut(R)->(Q::OpRep, Vec<L>) >
QtrajPoissonUnravel<R, C, Q, L, F>
where C::R : SampleUniform + Into<R> ,
      R: Into<C::R> + From<C::R> ,
{
    pub fn new( lind_op_fn: F  ) -> Self{
        let rand_uniform = Uniform::new(C::R::zero(), C::R::one());
        let rtol = NumCast::from(1.0e-2).unwrap();
        let atol = NumCast::from(1.0e-4).unwrap();
        return Self{ lind_op_fn, rand_uniform, rtol, atol, last_lind_ops: None, _phantom: PhantomData}
    }
    /// Set relative tolerance on norm loss (Default 1.0e-2)
    pub fn with_rtol(self, rtol: C::R) -> Self{
        return Self{rtol, ..self};
    }

    /// Set absolute tolerance on norm loss (Default 1.0e-4)
    pub fn with_atol(self, atol: C::R) -> Self{
        return Self{atol, ..self};
    }

    fn dpsi_dt(&mut self, t: R, psi: &Q::KetRep, psi2: &mut Q::KetRep) -> Result<(),()>{
        let (h0, lindops) = (self.lind_op_fn)(t);
        Q::khemv(&h0, C::one(), psi, psi2, C::zero() );
        let mut psi1 = psi.clone();
        psi1.scal(C::zero());
        for l in lindops.iter(){
            l.add_positive_map_to(psi, &mut psi1)
        }

        Q::kaxpy(C::from_subset(&-0.5), &psi1, psi2);
        Ok(())
    }
    ///
    /// g: Recording function for &ODEData before each step
    pub fn propagate_to_jump<RNG: Rng, G>
    (
            &mut self, t0: R, psi0: &Q::KetRep, dt: R, tf: R,
            rf: C::R, rng: &mut RNG, mut g: G)
    -> (Q::KetRep, JumpType<R, usize>)
    where C: From<R>+From<f64>,
          f64 : Into<R> + Into<C::R>,
          G: FnMut(&ODEData<R, Q::KetRep>) -> (),
    {
        use vec_ode::LinearCombination;
        use std::convert::From;
        use num_traits::real::Real;
        use qrs_core::util::iter::Sum;
        let atol = self.atol;
        let rtol = self.rtol;

        let f = |t: R, psi: &Q::KetRep, psi2: &mut Q::KetRep| {
            self.dpsi_dt(t, psi, psi2)
         };
        // Not bothering to preserve the norm
        let mut solver = RK45Solver::new_with_lc(f, t0, tf, psi0.clone(), dt, QRSLinearCombination::new())
            .with_tolerance((1.0e-8).into(), (1.0e-6).into());
        let mut prev_r = Q::ket_norm_sq(&solver.ode_data().x);
        let mut prev_x = psi0.clone();
        let mut prev_t = t0;
        loop {
            g(&solver.ode_data());
            let state = solver.step_adaptive();
            match state {
                ODEState::Ok(step) => {
                    let dt = match step{
                        ODEStep::Step(dt) => dt,
                        ODEStep::Reject | ODEStep::Chkpt => continue, // Disregard a rejection or checkpoint
                        _ => { panic!("Invalid ODEStep")
                        }
                    };

                    let r = Q::ket_norm_sq(&solver.ode_data().x);
                    let dr = prev_r - r;
                    let remaining_r: C::R = r - rf;
                    let tol : C::R = atol + rtol * remaining_r.max(C::R::zero());
                    // Check that we have not overshot rf
                    if r < rf - tol{
                        let ode_dat = solver.ode_data_mut();
                        ode_dat.x = prev_x.clone();
                        ode_dat.t = prev_t;
                        // If r < rf - tol here, but did not jump in the previous iteration,
                        // then dr > tol at this point
                        // Reduce the next step size more aggressively
                        let next_h : R =  (tol / dr.abs()).into() * dt * 0.5.into() ;
                        ode_dat.reset_step_size(next_h);
                        continue;
                    }
                    // Jump if r < rf
                    if remaining_r < C::R::zero(){
                        break;
                    }
                    // Make sure dr is small (relative to remaining r and absolutely to r_epsilon)
                    if dr.abs() > tol{
                        let next_h : R =  (tol.abs() / dr.abs()).into() * dt * 0.9.into();
                        solver.ode_data_mut().reset_step_size(next_h);
                    }
                    // Apply the observer function on the solver and continue to next iteration
                    prev_x = solver.ode_data().x.clone();
                    prev_t = solver.ode_data().t;
                    prev_r = r;
                }
                ODEState::Done => {
                    let r = Q::ket_norm_sq(&solver.ode_data().x);
                    let (tf, psif) = solver.into_current();
                    return (psif, JumpType::NoJump(tf));
                }
                ODEState::Err(e) => {
                    panic!(format!("ODE Error occured: {}", e.msg));
                }
            }

        }
        // Evaluate final state and ops
        let (tf, psif) = solver.into_current();
        let (_, lind_ops) = (self.lind_op_fn)(tf);

        // Evaluate probability vector
        let mut p_vec = lind_ops.iter().map(|l| l.positive_ev(&psif)).collect_vec();
        let p_tot = p_vec.iter().fold(C::R::zero(), |acc, &p| acc + p);
        for p in p_vec.iter_mut(){
            *p /= p_tot
        }
        // Convert to cumulative probabilities
        p_vec.iter_mut().fold(
            C::R::zero(),
            |acc, p| {
                let c = acc + *p;
                *p = c;
                c
            });
        // Sample a jump according to probabilities
        let x = rng.sample(&self.rand_uniform);

        let mut i_jmp = None;
        for (i,&p) in p_vec.iter().enumerate(){
            if x < p{
                i_jmp = Some(i);
                break
            }
        }
        let i_jmp = i_jmp.unwrap();
        let lind_i = &lind_ops[i_jmp];
        let mut psif_jmp = lind_i.map(&psif);
        let psif_norm = Q::ket_norm(&psif_jmp);
        Q::krscal(psif_norm.recip(), &mut psif_jmp);
        self.last_lind_ops = Some(lind_ops);
        return (psif_jmp, JumpType::Jump(tf,i_jmp));

    }
}

use crate::oqs::bath::Bath;
use num_complex::Complex64 as c64;
use qrs_core::reps::matrix;
use crate::oqs::adiabatic_me::AME;

/// Solves the adiabatic master equation constructed according to the provided
/// partitioned Hamiltonian, Lindblad operators, and bath spectral density
pub fn solve_aqt<B: Bath<f64>>(
    ame: &mut AME<B>,
    initial_state: matrix::Ket<c64>,
    tol: f64,
    dt_max_frac: f64,
    basis_tgts: Option<&Vec<usize>>,
    mut callback: Option<&mut dyn FnMut(&matrix::Ket<c64>)>
)
    -> Result<(matrix::Ket<c64>), vec_ode::ODEError>
{
    use log::info;
    use crate::oqs::adiabatic_me::basis_tgts_ampls;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    // Important variables
    let partitions  = ame.adb.haml.partitions().to_owned();
    let num_partitions = partitions.len() - 1;
    let n = ame.adb.haml.basis_size() as usize;
    let t0 = partitions[0];
    let mut rng = Xoshiro256Plus::from_entropy();
    let rand_uniform = Uniform::new(0.0, 1.0);
    let mut norm_est = condest::Normest1::<f64>::new(n, 2);
    let mut psi0 = initial_state;
    let mut last_delta_t : Option<f64> = None;

    callback.as_deref_mut().map(|f| f(&psi0));
    //results_vec.push(ame_state);

    let mut prev_r = None;
    // Iterate over each partition
    for (p, (&ti0, &tif)) in partitions.iter()
        .zip(partitions.iter().skip(1))
        .enumerate()
    {
        let delta_t = tif - ti0;
        let max_step = dt_max_frac * delta_t;
        let min_step = max_step * 1.0e-12;
        let dt = match last_delta_t{
            None => dt_max_frac * delta_t / 20.0,
            Some(dt) => dt.max(1.001*min_step).min(0.999*max_step) };

        info!("Evolving partition {} / {} \t ({}, {})", p+1, num_partitions,
              ti0, tif);
        // Prepare evolution for partition p
        ame.load_partition(p);
        // Evaluation of dpsi/dt

        // Need to share and mutate over two lambdas
        let last_solver_dt = std::cell::Cell::new(dt);
        let solver_step_t0 = std::cell::Cell::new(0.0);
        let solver_step_tf = std::cell::Cell::new(dt);
        let f = |t: f64|{
            let dt = last_solver_dt.get();
            let t0 = solver_step_t0.get();
            let tf = solver_step_tf.get();
            ame.adb.step_eigvs(t0, tf);
            ame.generate_sparse_lindops(t, last_solver_dt.get() * 0.25)
        };
        // Function to evaluate before each attempted step
        // Updates the step size and time interval for ame.adb evaluation
        let g = |ode_data: &ODEData<f64, matrix::Ket<c64>>|{
            let h = ode_data.step_size().unwrap_dt_or(last_solver_dt.get());
            let t0 = ode_data.t;
            let tf = t0 + h;
            last_solver_dt.set(h);
            solver_step_t0.set(t0);
            solver_step_tf.set(tf);
        };
        let mut qt = QtrajPoissonUnravel::<f64, c64, matrix::DenseQRep<c64>, _, _>::new(f);

        //  ** Begin waiting time algorithm **
        // For time range [ti0, tif]
        let mut t = ti0;

        while t < tif {
            let rf = prev_r.unwrap_or(rng.sample(rand_uniform));
            let (psif, jmp) = qt.propagate_to_jump(t, &psi0, last_solver_dt.get(), tif, rf,  &mut rng,  g);
            psi0 = psif;
            t = jmp.t();
            if let JumpType::Jump(_, i) = jmp {
                println!("t = {}, Jump {} ", t, i);
                prev_r = None;
            } else {
                // If propagation terminates with no jump, continue from the last r
                prev_r = Some(rf)
            }
        }
        last_delta_t = Some(last_solver_dt.get());
        // Callback on wave function after each time range
        callback.as_deref_mut().map(|f| f(&psi0));
    }

    Ok(psi0)
}

#[cfg(test)]
mod tests{
    use num_complex::{Complex64 as c64};
    use qrs_core::reps::dense::*;
    use qrs_core::quantum::LinearOperator;
    use rand_xoshiro::Xoshiro256Plus;
    use crate::base::pauli::dense::*;
    use crate::ComplexScalar;
    use super::{QtrajPoissonUnravel, JumpType};
    use rand::{SeedableRng, Rng};
    use rand_distr::Uniform;

    /// Represents the sparse linear operator
    ///  g | i > < j |
    #[derive(Copy, Clone)]
    struct SparseLinOp{
        pub i: u32,
        pub j: u32,
        pub g: f64
    }
    
    impl LinearOperator<Ket<c64>, c64> for SparseLinOp{
        fn map(&self, v: &Ket<c64>) -> Ket<c64> {
            let mut lv = Ket::zeros(v.raw_dim());
            unsafe { *lv.uget_mut(self.i as usize) = 
                v.uget(self.j as usize) * self.g; }
            return lv;
        }

        fn conj_map(&self, v: &Ket<c64>) -> Ket<c64> {
            let mut lv = Ket::zeros(v.raw_dim());
            unsafe { *lv.uget_mut(self.j as usize) =
                v.uget(self.i as usize) * self.g; }
            return lv;
        }
        
        fn positive_map(&self, v: &Ket<c64>) -> Ket<c64>{
            let mut lv = Ket::zeros(v.raw_dim());
            unsafe{
                *lv.uget_mut(self.j as usize) =
                    v.uget(self.j as usize) * self.g*self.g;
            }
            return lv;
        }

        fn add_map_to(&self, v: &Ket<c64>, target: &mut Ket<c64>) {
            unsafe{
                *target.uget_mut(self.i as usize) +=
                    v.uget(self.j as usize) * self.g;
            }
        }

        fn add_conj_map_to(&self, v: &Ket<c64>, target: &mut Ket<c64>) {
            unsafe{
                *target.uget_mut(self.j as usize) +=
                    v.uget(self.i as usize) * self.g;
            }
        }

        fn add_positive_map_to(&self, v: &Ket<c64>, target: &mut Ket<c64>) {
            unsafe{
                *target.uget_mut(self.j as usize) +=
                    v.uget(self.j as usize) * (self.g*self.g);
            }
        }

        fn positive_ev(&self, v: &Ket<c64>) -> f64 {
            use qrs_core::ComplexField;
            unsafe {
                return (self.g * self.g) * v.uget(self.j as usize).norm_sqr();
            }
        }
    }

    fn eval_time_indp_lind(_t: f64, gamma: f64, beta: f64) -> (Op<c64>, Vec<SparseLinOp>){
        let h = sz() * ( -c64::i() );
        let l1 = SparseLinOp{i: 0, j: 1, g: gamma};
        let l2 = SparseLinOp{i: 1, j: 0, g: f64::exp(-2.0 * beta)*gamma};
        let lind_ops = vec![l1, l2];
        return (h, lind_ops)
    }
    #[test]
    fn test_qtraj(){
        let beta = 1.0;
        let gamma = 2.0;
        let f = |t| eval_time_indp_lind(t, gamma, beta);
        let mut rng = Xoshiro256Plus::from_entropy();
        let rand_uniform = Uniform::new(0.0, 1.0);
        let mut qt = QtrajPoissonUnravel::<f64, c64, DenseQRep<c64>, _, _>::new(f );
        let mut psi0 = Ket::from(vec![1.0.into(), 0.0.into()]);
        let dt = 0.01;
        let tf = 1000.0;
        let mut t = 0.0;
        let r_epsilon = 1.0e-4;

        while t < tf {
            let rf = rng.sample(rand_uniform);
            let (psif, jmp) = qt.propagate_to_jump(t, &psi0, dt, tf, rf,  &mut rng,  |_|{});
            psi0 = psif;
            t = jmp.t();
            if let JumpType::Jump(_, i) = jmp{
                println!("t = {}, Jump {} ", t, i)
            }

        }
        println!("{}", psi0)
    }
}
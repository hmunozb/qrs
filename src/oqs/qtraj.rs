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
    _phantom: PhantomData<(Q, R, L)>
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
        return Self{ lind_op_fn, rand_uniform, rtol, atol, _phantom: PhantomData}
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
    /// g: Recording function for &ODEData after each accepted step
    pub fn propagate_to_jump<RNG: Rng, G>
    (
            &mut self, t0: R, psi0: &Q::KetRep, dt: R, tf: R,
            rf: C::R, rng: &mut RNG, mut g: G)
    -> (Q::KetRep, JumpType<R, usize>)
    where C: From<R>+From<f64>,
          f64 : Into<R>,
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
        let mut solver = RK45Solver::new_with_lc(f, t0, tf, psi0.clone(), dt, QRSLinearCombination::new());
        let mut prev_r = Q::ket_norm_sq(&solver.ode_data().x);
        let mut prev_x = psi0.clone();
        let mut prev_t = t0;
        loop {
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
                        let next_h : R = 0.5.into() * (tol / dr.abs()).into() * dt;
                        ode_dat.reset_step_size(next_h);
                        continue;
                    }
                    // Jump if r < rf
                    if remaining_r < C::R::zero(){
                        break;
                    }
                    // Make sure dr is small (relative to remaining r and absolutely to r_epsilon)
                    if dr.abs() > tol{
                        let next_h : R = 0.9.into() * (tol.abs() / dr.abs()).into() * dt;
                        solver.ode_data_mut().reset_step_size(next_h);
                    }
                    // Apply the observer function on the solver and continue to next iteration
                    g(solver.ode_data());
                    prev_x = solver.ode_data().x.clone();
                    prev_t = solver.ode_data().t;
                    prev_r = r;
                }
                ODEState::Done => {
                    let r = Q::ket_norm_sq(&solver.ode_data().x);
                    g(solver.ode_data());
                    let (tf, psif) = solver.into_current();
                    return (psif, JumpType::NoJump(tf));
                }
                ODEState::Err(e) => {
                    panic!(format!("ODE Error occured: {}", e.msg));
                }
            }

        }
        // Evaluate final state and ops
        g(solver.ode_data());
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

        return (psif_jmp, JumpType::Jump(tf,i_jmp));

    }
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
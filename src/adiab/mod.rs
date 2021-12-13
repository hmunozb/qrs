use crate::util::degen::{degeneracy_detect, handle_degeneracies,
                         handle_degeneracies_relative, handle_degeneracies_relative_vals,
                         handle_relative_phases};
use qrs_core::RealScalar;
use qrs_timed::timed_op::TimeDependentOperator;
use crate::util::TimePartHaml;
use crate::util::TimeDepMatrix;
use vec_ode::LinearCombination;
use qrs_core::reps::matrix::{Ket, Op, LC};
use num_complex::Complex64 as c64;
use num_complex::Complex;
use log::{trace, warn};

fn assert_orthogonal<T>(v:& Op<Complex<T>>)
    where T: RealScalar
{
    let vad = v.adjoint();
    let vv : Op<Complex<T>> = vad * v;

    assert!(vv.is_identity(T::from_subset(&1.0e-9)), "Orthogonal assertion failed");
}

/// Evaluates the diabatic perturbation  -K_A(t)  to second order with spacing dt
    /// In the adiabatic frame,
    ///  K_A = i W^{dag} (dW/dt)
pub fn diabatic_driver(t: f64, p: usize, dt: f64, haml: &mut TimePartHaml<f64>,
                        eigvals: & Ket<f64>,
                        eigvecs: & Op<c64>,
                        diab_k: &mut Op<c64>
){
    //let n = DenseQRep::qdim_op(&self.adiab_haml);
    let (_vals1, vecs1) = haml.eig_p(t-dt, Some(p));
    let (_vals2, vecs2) = haml.eig_p(t+dt, Some(p));

    let mut w1 = eigvecs.ad_mul(&vecs1);
    let mut w2 = eigvecs.ad_mul(&vecs2);
    // Degeneracies have to be handled carefully for the diabatic perturbation
    let degens = degeneracy_detect(eigvals, None);

    if degens.len() > 0{
        handle_degeneracies(&degens, &mut w1);
        handle_degeneracies(&degens, &mut w2);
    }

    let mut w_dot = w2;
    w_dot -= &w1;

    LC::scalar_multiply_to(&w_dot, -Complex::<f64>::i() * Complex::from(1.0/(2.0 * dt)),
                           diab_k);

    // Ensure K_A is hermitian
    let mut temp = w1;
    diab_k.adjoint_to(&mut temp);
    *diab_k += &temp;
    *diab_k /= Complex::from(2.0);

}

/// This structure is responsible for keeping track of eigenvalues and eigenvectors
/// for an adiabatic Hamiltonian specified by a TimePartHaml
///
pub struct AdiabaticPartitioner<'a>{
    pub haml: &'a TimePartHaml<'a, f64>,
    pub eigvecs: Op<c64>,
    pub eigvals: Ket<f64>,
    pub interval_t: Option<(f64, f64)>,
    pub interval_eigvecs: (Op<c64>, Op<c64>),

    p: usize,
    ztemp: Op<c64>,
}

impl<'a> AdiabaticPartitioner<'a>{
    /// Initializes at partition 0
    pub fn new(haml: &'a TimePartHaml<'a, f64>,) -> Self{
        let n = haml.basis_size() as usize;
        let p = 0;

        Self{
            haml,
            eigvecs: Op::zeros(n, n),
            eigvals: Ket::zeros(n),
            interval_t: None,
            interval_eigvecs: (Op::zeros(n, n), Op::zeros(n, n)),
            p: 0,
            ztemp: Op::zeros(n, n),
        }
    }

    /// Load the partition p from the time-partitioned Hamiltonian
    /// Currently, will panic if p + 1 != self.p, when p > 0.
    ///
    /// This uses the partition projectors of the Hamiltonian to project
    /// the interval eigenvectors onto partition p (epi), then directly diagonalizes
    /// the interval in the basis of partition p (evi), and then ensures that evi is adiabatically
    /// normal with respect to epi.
    pub fn load_partition(&mut self, p: usize){
        // self.p_lindblad_ops.clear();
        match self.interval_t {
            None => {},
            Some((t0, tf)) =>{
                if p > 0 {
                    assert_eq!(p, self.p+1);
                    let proj = self.haml.projector(p-1);
                    let ep0 = proj.ad_mul(&self.interval_eigvecs.0);
                    let epf = proj.ad_mul(&self.interval_eigvecs.1);

                    let (vals0, mut ev0) = self.haml.eig_p(t0, Some(p));
                    let (valsf, mut evf) = self.haml.eig_p(tf, Some(p));
                    handle_degeneracies_relative_vals(&vals0, &mut ev0, &ep0, None);
                    handle_degeneracies_relative_vals(&valsf, &mut evf, &epf, None);
                    let mut rank_loss = handle_relative_phases(&mut ev0, &ep0, &mut self.ztemp);
                    rank_loss += handle_relative_phases(&mut evf, &epf, &mut self.ztemp);
                    if rank_loss > 0{
                        warn!("{} adiabatic rank(s) scattered going into partition {}", rank_loss, p);
                    }
                    self.interval_eigvecs.0 = ev0;
                    self.interval_eigvecs.1 = evf;
                }
            }
        }
        self.p = p;
    }

    pub fn to_current_basis(&self, op: &Op<c64>) -> Op<c64>{
        self.haml.transform_to_partition(op, self.p )
    }

    /// Evaluates the eigenvalues and eigenvectors of time t at the current partition
    /// No gauge or degeneracy handling
    pub fn eigv(& self, t: f64) -> (Ket<f64>, Op<c64>){
        let (vals, vecs) =
            self.haml.eig_p(t, Some(self.p ));
        (vals, vecs)
    }

    /// Evaluates the eigenvalues and eigenvectors of time t at the current partition
    /// and stores them in self.eigvals and self.eigvecs
    pub fn load_eigv(&mut self, t: f64){
        let (eigvals, eigvecs) = self.normalized_eigv(t);
        self.eigvals = eigvals;
        self.eigvecs = eigvecs;
    }

    /// Loads the eigenvalues and eigenvectors of time t at the current partition
    /// with degeneracy handling
    pub fn eigv_degen_handle(&self, t: f64, v0: &Op<c64>) -> (Ket<f64>, Op<c64>){
        let (vals, mut vecs) =
            self.haml.eig_p(t, Some(self.p ));

        let degens = degeneracy_detect(&vals, None);
        if degens.len() > 0{
            handle_degeneracies_relative(&degens, &mut vecs,v0);
            assert_orthogonal(& vecs);
            trace!("A degeneracy at time {} was handled. ", t)
        }

        (vals, vecs)
    }

    /// Evaluates the diabatic perturbation  -K_A(t)  to second order with spacing dt
    /// In the adiabatic frame,
    ///  K_A = i W^{dag} (dW/dt)
    /// The eigenvectors are evaluated with normal phase and degeneracy to ensure
    /// numerical stability
    ///
    /// Also stores (vals, vecs) evaluated at t into self.eigvals, self.eigvecs
    pub fn diabatic_driver(&mut self, t: f64, dt: f64, diab_k: &mut Op<c64>){

        let (eigvals, eigvecs) = self.normalized_eigv(t);
        let (_vals1, vecs1) =  self.normalized_eigv(t-dt);
        let (_vals2, vecs2) = self.normalized_eigv(t+dt);

        let w1 = eigvecs.ad_mul(&vecs1);
        let w2 = eigvecs.ad_mul(&vecs2);

        let mut w_dot = w2;
        w_dot -= &w1;

        LC::scalar_multiply_to(&w_dot, -Complex::<f64>::i() * Complex::from(1.0/(2.0 * dt)),
                               diab_k);

        // Ensure K_A is hermitian
        let mut temp = w1;
        diab_k.adjoint_to(&mut temp);
        *diab_k += &temp;
        *diab_k /= Complex::from(2.0);

        self.eigvals = eigvals;
        self.eigvecs = eigvecs;
    }

    fn reset_eigvs(&mut self, t0: f64, tf: f64){
        let (_, v0) = self.eigv(t0);
        let (_, mut vf) = self.eigv_degen_handle(tf,  &v0);
        handle_relative_phases(&mut vf, &v0, &mut self.ztemp);
        self.interval_eigvecs.0 = v0;
        self.interval_eigvecs.1 = vf;
    }

    /// Changes the state of the partitioner to the time interval (t0, tf)
    /// The eigenvectors are changed in a way such that if tf == t0, then
    /// v0 <- vf. Furthermore, vf is loaded in a numerically stable manner
    /// by handling degeneracies and relative phases.
    pub fn step_eigvs(&mut self, t0: f64, tf: f64) {
        let prev_t = self.interval_t;
        //let (v0, vf) =
        match prev_t{
            None =>{
                self.reset_eigvs(t0, tf);
            }
            Some((s0, sf)) =>{
                if (s0 == t0) && (sf == tf) { // Unlikely but simple case of the exact same interval
                    ( )
                } else if s0 == t0 { // Solver rejected tf and is attempting a smaller time step
                    let v0 = &self.interval_eigvecs.0;
                    // change final eigvec only
                    let (_, mut vf) = self.eigv_degen_handle(tf,  &v0);
                    handle_relative_phases(&mut vf, v0, &mut self.ztemp);
                    self.interval_eigvecs.1 = vf;
                } else if t0 == sf { // Solver accepted the previous step
                    // Previous final eigvec is now the current initial eigvec
                    std::mem::swap(&mut self.interval_eigvecs.0, &mut self.interval_eigvecs.1);
                    let v0 = & self.interval_eigvecs.0;
                    let (_, mut vf) = self.eigv_degen_handle(tf,  &v0);
                    handle_relative_phases(&mut vf, v0, &mut self.ztemp);
                    // set new final eigvec
                    self.interval_eigvecs.1 = vf;
                } else if sf == tf { // very unlikely case that initial eigvec must be set
                    // For now, this simply behaves like the time reversal of the rejection case
                    warn!("generate_split: An generally impossible branch was reached. (sf==tf)");
                    let vf = &self.interval_eigvecs.1;
                    let (_, mut v0) = self.eigv(t0);
                    handle_relative_phases(&mut v0, vf, &mut self.ztemp);
                    self.interval_eigvecs.0 = v0;
                } else {
                    self.reset_eigvs(t0, tf);
                }
            }
        };

        self.interval_t = Some((t0, tf));
    }

    pub fn eigv_interval(&self) -> (&Op<c64>, &Op<c64>){
        (&self.interval_eigvecs.0, &self.interval_eigvecs.1)
    }

    /// Load the eigenvalues and normal eigenvectors
    /// which are degeneracy and phase fixed relative to v0
    pub fn normalized_eigv(&mut self, t: f64) -> (Ket<f64>, Op<c64>){
        let v0 = &self.interval_eigvecs.0;
        let (vals, mut vecs) = self.eigv_degen_handle(t,  v0);
        handle_relative_phases(&mut vecs, v0, &mut self.ztemp);

        (vals, vecs)

    }

}
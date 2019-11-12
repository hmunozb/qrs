use crate::util::*;
use crate::base::quantum::{QRep};
use crate::base::dense::*;
use crate::oqs::bath::Bath;
use crate::ode::dense::DenseExpiSplit;
use crate::ode::super_op::{KineticExpSplit, CoherentExpSplit, DenMatExpiSplit};
use alga::general::{ComplexField, RealField};
use blas_traits::BlasScalar;
use num_traits::{Zero, Float};
use num_complex::Complex;
use num_complex::Complex64 as c64;
use nalgebra::{DVector, DMatrix};
use vec_ode::exp::split_exp::{SemiComplexO4ExpSplit, CommutativeExpSplit, TripleJumpExpSplit};
use itertools::Itertools;
use vec_ode::exp::{DirectSumL, ExponentialSplit};
use ndarray_stats::*;
use ndarray::{ArrayBase, Array, Array1, Array2, ArrayView2, ShapeBuilder};
use vec_ode::{ODEState, ODEStep, ODESolver, ODESolverBase};
use vec_ode::exp::cfm::ExpCFMSolver;
use vec_ode::AdaptiveODESolver;

//use alga::linear::NormedSpace;

type AMEDissipatorSplit<T> = CommutativeExpSplit<T, Complex<T>, Op<T>,
                                KineticExpSplit<T>,
                                CoherentExpSplit>;
type AdiabaticMEExpSplit<T> = TripleJumpExpSplit<T, Complex<T>, Op<T>,
                                DenMatExpiSplit<T>,
                                AMEDissipatorSplit<T>>;

type AdiabaticMEL<T> = <AdiabaticMEExpSplit<T> as ExponentialSplit<T, Complex<T>, Op<T>>>::L;

fn make_ame_split<T: RealField + Float>(n: u32) -> AdiabaticMEExpSplit<T>
where Complex<T> : BlasScalar + ComplexField<RealField=T>
{
    let split = AdiabaticMEExpSplit::<T>::new(
        DenMatExpiSplit::new(n),
        AMEDissipatorSplit::new(
             KineticExpSplit::new(n),
             CoherentExpSplit::new(n),)
    );


    split
}

struct AMEWorkpad<R: RealField>{
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

impl<R: RealField> AMEWorkpad<R>{

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
                ztemp0: DMatrix::zeros(n,n)}
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
fn ame_liouvillian<R: RealField, B: Bath<R>>(
    bath: & B,
    work: &mut AMEWorkpad<R>,
    vals: &DVector<R>,
    vecs: &DMatrix<Complex<R>>,
    haml: &mut DMatrix<Complex<R>>,
    lind_pauli: &mut DMatrix<Complex<R>>,
    lind_coh: &mut DMatrix<Complex<R>>,
    lindblad_ops: &Vec<DMatrix<Complex<R>>>
)
where Complex<R> : ComplexField<RealField=R>+BlasScalar
{
    assert_eq!(lindblad_ops.len(), work.linds_ab.len(), "Number of lindblad operators mismatched");
    let one_half = R::from_subset(&(-0.5_f64));

    //                  TRANSITION FREQUENCIES AND HAMILTONIAN
    // omega[i,j] = vals[i] - vals[j]
    outer_zip_to(vals, vals, &mut work.omega, |a, b| *a - *b);
    copy_transmute_to(& work.omega, haml);
    //H[a,b] = - i omega[a,b]
    DenseQRep::qscal(-Complex::i(), haml);

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
    //  The quantity sum_{alpha} abs( A[a,b] ).^2 is currently stored in lind_pauli
    //  and can be use to evaluate the lamb_shift
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

    // Back to Pauli rates
    lind_pauli.zip_apply(&work.gamma,
                               |s, g| s*Complex::from(g));

    lind_pauli.fill_diagonal(Complex::zero());

    //Out_gamma =sum_a g_ab Asq[a,b] = sum_a tilde{Gamma}[a,b]
    for (g1, g0) in lind_pauli.column_iter()
            .zip(work.gamma_out.iter_mut()){
       *g0 = g1.sum();
    }

    // Finally
    // pauli_mat = E_Gamma_offdiag - diag(Out_rate)
    for i in 0..lind_pauli.nrows(){
        lind_pauli[(i,i)] =  -work.gamma_out[i];
    }

    //                      LINDBLAD COHERENT MATRIX EVALUATION
    //  Sum of two terms:
    // Lind_coh_1 = -(1.0 / 2.0) * (Out_rate[:, np.newaxis] + Out_rate[np.newaxis, :])
    outer_zip_to(&work.gamma_out, &work.gamma_out, &mut work.lind_coh_1,
        |a, b|  Complex::from(one_half) * ( a + b));

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

    // Lind_coh_2 = G0[a, b]* - (1/2)(G0[a,a] + G0[b,b])
    let n = work.lind_coh_2.ncols();
    for i in 0..n{
        for j in 0..n{
            work.lind_coh_2[(i,j)] = work.diag_gamma_0[(j,i)]
                - Complex::from(one_half) * (
                    work.diag_gamma_0[(i,i)] + work.diag_gamma_0[(j,j)] )
        }
    }

    lind_coh.copy_from(&work.lind_coh_1);
    *lind_coh += &work.lind_coh_2;
    lind_coh.fill_diagonal(Complex::zero());
}

pub struct AMEResults{
    t: Array1<f64>,
    rho: Array1<Op<f64>>,
    partitions: Vec<u32>,
    observables: Vec<Vec<c64>>
}

pub struct AME<'a, B: Bath<f64>>{
    haml: &'a TimePartHaml<'a, f64>,
    lindblad_ops: &'a Vec<Op<f64>>,
    p_lindblad_ops: Vec<Op<f64>>,
    work: AMEWorkpad<f64>,
    adiab_haml: Op<f64>,
    diab_k: Op<f64>,
    lind_pauli: Op<f64>,
    lind_coh: Op<f64>,
    eigvecs: Op<f64>,
    bath: &'a B
}

impl<'a, B: Bath<f64>> AME<'a, B> {
//    pub fn from_operators(haml_mat: TimeDepMatrix<'a, Complex<f64>>,
//                          lindblad_ops: &'a Vec<Op<f64>>){
//        let haml = TimePartHaml::new(TimeDepMatrix,)
//    }

    pub fn new(haml: &'a TimePartHaml<'a, f64>,
               lindblad_ops: &'a Vec<Op<f64>>,
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
            bath};
        me.load_partition(0);
        me
    }

    /// Replace the lindblad operators with the truncated basis of the
    /// pth partition
    /// (todo: should be modified when adaptive basis sizes are available)
    fn load_partition(&mut self, p: usize){
        self.p_lindblad_ops.clear();
        for lind in self.lindblad_ops{
            self.p_lindblad_ops.push(self.haml.transform_to_partition(lind, p ));
        }
    }

    pub fn adiabatic_lind(&mut self, t: f64, p: usize){
        let (vals, vecs) =
            self.haml.eig_p(t, Some(p ));
        self.eigvecs = vecs;
        ame_liouvillian(self.bath, &mut self.work, &vals, &self.eigvecs,
                        &mut self.adiab_haml, &mut self.lind_pauli, &mut self.lind_coh,
                        &self.p_lindblad_ops );
    }

    /// Evaluates the diabatic perturbation K_A(t) to second order with spacing dt
    /// In the adiabatic frame,
    ///  i W^{dag} (dW/dt)
    pub fn diabatic_driver(&mut self, t: f64, p: usize, dt: f64){
        let n = DenseQRep::qdim_op(&self.adiab_haml);
        let (vals1, vecs1) = self.haml.eig_p(t-dt, Some(p));
        let (vals2, vecs2) = self.haml.eig_p(t+dt, Some(p));
        let mut w_dot = vecs2;
        w_dot -= vecs1;
        w_dot *= Complex::from(1.0/(2.0 * dt));

        self.eigvecs.ad_mul_to(&w_dot, &mut self.diab_k);
        self.diab_k *= Complex::i();
        self.diab_k.adjoint_to(&mut self.work.ztemp0);
        self.diab_k += &self.work.ztemp0;
        self.diab_k /= Complex::from(2.0);

    }

    pub fn generate_split(&mut self, t_arr: &[f64], p: usize) -> Vec<AdiabaticMEL<f64>> {

        let dt_vec : Vec<f64> = t_arr.iter().zip(t_arr.iter().skip(1))
            .map(|(a, b)| *b - *a).collect_vec();
        let dt_arr = Array1::from(dt_vec);
        let dt_mean = (dt_arr).mean().unwrap();

        let mut l_vec = Vec::new();
        for &t in t_arr{
            self.adiabatic_lind(t, p);
            let mut coh_split_l = self.lind_coh.clone();
            coh_split_l += &self.adiab_haml;
            let dsl_lind = DirectSumL::new(self.lind_pauli.clone(), coh_split_l);
            self.diabatic_driver(t, p, dt_mean);
            let dsl = DirectSumL::new(self.diab_k.clone(), dsl_lind);
            l_vec.push(dsl);
        }

        l_vec
    }



}

pub fn solve_ame<B: Bath<f64>>(ame: &mut AME<B>, initial_state: Op<f64>, tol: f64) -> Op<f64>{
    let partitions  = ame.haml.partitions().to_owned();
    let n = ame.haml.basis_size() as usize;
    let mut norm_est = condest::Normest1::new(n, 2);
    let mut rho0 = initial_state;

    for (p, (&ti0, &tif)) in partitions.iter()
            .zip(partitions.iter().skip((1)))
            .enumerate()
    {
        let delta_t = tif - ti0;
        let dt = delta_t/200.0;


        ame.load_partition(p);
        let split = make_ame_split(n as u32);

        let f = |t_arr: &[f64]|{
                ame.generate_split(t_arr, p)
            };
        let norm_fn = |m: &Op<f64>|{
            //should technically be transposed to row major but denmats are hermitian
            let arr = ArrayView2::from_shape((n, n), m.as_slice()).unwrap();
                norm_est.normest1(&arr, 5)
            };
        let mut rhodag = rho0.clone();
        let mut solver = ExpCFMSolver::new(
            f, norm_fn,ti0, tif, rho0,  dt, split)
            .with_tolerance(1.0e-6, tol)
            .with_step_range(dt*1.0e-4,
                             dt*10.0)
            .with_init_step(dt);
        while let ODEState::Ok(step) = solver.step(){
            //Enforce Hermiticity after each step
            let mut dat = solver.ode_data_mut();
            dat.x.adjoint_to(&mut rhodag);
            dat.x += &rhodag;
            dat.x /= c64::from(2.0);
        }

        let(_, rhof) = solver.into_current();
        rho0 = rhof;
    }

    rho0
}

#[cfg(test)]
mod tests{
    use num_complex::Complex;
    use super::*;
    use crate::base::quantum::QRep;
    use crate::base::dense::*;
    use crate::base::pauli::dense as pauli;
    use crate::oqs::bath::OhmicBath;
    use crate::ode::super_op::DenMatExpiSplit;
    use crate::util::{TimeDepMatrixTerm, TimeDepMatrix, TimePartHaml};
    use crate::oqs::adiabatic_me::{AME, solve_ame, make_ame_split};
    use alga::general::RealField;
    use num_complex::Complex64 as c64;
    use num_traits::real::Real;
    use vec_ode::LinearCombination;

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

        let mut split = make_ame_split(2);
        let mut spl1 = ame.generate_split(&[0.50*tf, 0.51*tf], 0);
        spl1[0].scale(c64::from(0.1));

        let spu1 = split.exp(&spl1[0]);

        let rho0 = (id.clone() + &sy)/c64::from(2.0);
        let rho1 = split.map_exp(&spu1, &rho0);

        println!("Initial adiabatic density matrix:\n{}", rho0);
        println!("Action at tf/2:\n{}", rho1);

        let rhof = solve_ame(&mut ame, rho0, 1.0e-6);

        println!("Final density matrix:\n{}", rhof);
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

        let tf = 2000.0;
        let dt = 0.01;
        let mut sp_haml = DenMatExpiSplit::<f64>::new(2);
        let id = pauli::id::<f64>();
        let sx = pauli::sx::<f64>();
        let sy = pauli::sy::<f64>();
        let sz = pauli::sz::<f64>();

        let fx = |t: f64| Complex::from(1.0*(t/tf).cos());
        let fy = |t: f64| Complex::from(1.0*(t/tf).sin());

        let hx = TimeDepMatrixTerm::new(&sx, &fx);
        let hy = TimeDepMatrixTerm::new(&sy, &fy);
        let lz = sz.clone();
        let lind_ops = vec![lz];


        let haml = TimeDepMatrix::new(vec![hx, hy]);
        let haml_part = TimePartHaml::new(haml, 2, 0.0, tf,1);

        let mut ame = AME::new(&haml_part, &lind_ops, &bath);

        let rho0 = (id.clone() + sz)/c64::from(2.0);
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

        let rhof = solve_ame(&mut ame, rho0, 1.0e-6);

        println!("Final density matrix:\n{}", rhof);
    }
}
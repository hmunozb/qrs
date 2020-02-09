use crate::util::simd_phys::vf64::{Aligned4xf64};
use crate::util::simd_phys::r3::{Vector3d4xf64, Matrix3d4xf64};

use ndarray::{Axis, Array2, ArrayView1, ArrayViewMut1, ArrayView2, ArrayView3};
//use nalgebra::{Vector3, Matrix3, Matrix};
use num_traits::Zero;
use rand::Rng;
use itertools;
//use simd_phys::aligned::Aligned4x64;
use simd_phys::r3::cross_exponential_vector3d;
use nalgebra::Vector3;
//use simd_phys::aligned::Aligned4x64;

pub type SpinVector3DAligned4xf64 =  Vector3d4xf64;
//type SpinArray3DAligned4x64 = [Aligned4xf64; 3];

//static ZERO_SPIN_ARRAY_3D: SpinArray3DAligned4x64 = [Aligned4xf64{dat: [0.0, 0.0, 0.0, 0.0]}; 3];


pub fn xyz_to_array_chunks(arr: ArrayView2<f64>,
                       mut chunk_array: ArrayViewMut1<SpinVector3DAligned4xf64>) {
    let shape = arr.shape();
    if shape[1] != 3{
        panic!("xyz_to_array_chunks: 3 spatial dimensions required.");
    }
    let n = shape[0];
    let n_ch = (n-1)/4 + 1;
    if chunk_array.shape()[0] != n_ch{
        panic!("xyz_to_array_chunks: mismatching chunk size")
    }

    for (xyz_chunk, mut chunk_4xf64) in ArrayView2::axis_chunks_iter(&arr,Axis(0), 4)
            .zip(chunk_array.iter_mut())
    {
        //xyz_chunk is a 4 x 3D view, while chunk_4xf64 is a a single 3D x 4xf64 array

        //transpose is now 3D x 4
        let xyz_chunk_t = xyz_chunk.t();
        for (x1, x2) in xyz_chunk_t.genrows().into_iter().zip(chunk_4xf64.iter_mut()){
            for (&x1i, x2i) in x1.iter().zip(x2.dat.iter_mut()){
                *x2i = x1i;
            }
        }
    }

}

/// Evaluates v in the dynamical spin-langevin equation
///  dm/dt = v \cross m
/// where
///     v =  h + \chi (h \cross m) )
/// Specifically, this function updates the hamiltonian field by adding the dissipative term
///     h += \chi (h\cross m)
///
/// h: Hamiltonian local fields for each spin
/// m: the 3D rotor spin
///
/// The arrays are passed as arrays of 3_D x 4_vf64 chunks
fn sl_add_dissipative(
    h_array: &mut ArrayViewMut1<SpinVector3DAligned4xf64>,
    m_array: & ArrayView1<SpinVector3DAligned4xf64>,
    chi: f64
){
//    let h_shape = h_array.raw_dim();
//    let spins_shape = m_array.raw_dim();
//    let dm_shape = v_array.raw_dim();
//    if h_shape != spins_shape{
//        panic!(format!("spin_langevin_dmdt: Dimension mismatch: h={:?} , m={:?} ", h_shape, spins_shape))
//    }
//
//    if dm_shape != spins_shape{
//        panic!(format!("spin_langevin_dmdt: Dimension mismatch: m={:?} , dm={:?} ", spins_shape, dm_shape))
//    }
    let chi = Aligned4xf64::from(chi);
    for (m,h) in m_array.iter()
            .zip(h_array.iter_mut()){
        //let mut dh = SpinArray3DAligned4x64::default();
        //cross_product(&*h, m, &mut dh);
        let dh = h.cross(m);
        *h += dh * chi;
//        for (dhi,hi) in dh.iter().zip(h.iter_mut()){
//            *hi +=  *dhi * chi;
//        }
    }
}

pub struct SpinLangevinWorkpad{
    m0: Array2<SpinVector3DAligned4xf64>,
    h0: Array2<SpinVector3DAligned4xf64>,
    h1: Array2<SpinVector3DAligned4xf64>,
    h2: Array2<SpinVector3DAligned4xf64>,
    m1: Array2<SpinVector3DAligned4xf64>,
    omega1: Array2<SpinVector3DAligned4xf64>,
    omega2: Array2<SpinVector3DAligned4xf64>,
    chi1: Array2<SpinVector3DAligned4xf64>,
    chi2: Array2<SpinVector3DAligned4xf64>

}

impl SpinLangevinWorkpad{
    pub fn from_shape(s0: usize, s1: usize) -> Self{
        let sh = (s0, s1);
        Self{
            m0: Array2::from_elem(sh,Zero::zero()),
            h0: Array2::from_elem(sh, Zero::zero()), h1: Array2::from_elem(sh, Zero::zero()), h2:  Array2::from_elem(sh, Zero::zero()),
            m1:  Array2::from_elem(sh,Zero::zero()), omega1:  Array2::from_elem(sh,Zero::zero()), omega2:  Array2::from_elem(sh,Zero::zero()),
            chi1: Array2::from_elem(sh,Zero::zero()), chi2: Array2::from_elem(sh,Zero::zero())
        }
    }

    pub fn shape(&self) -> (usize, usize){
        let sh = self.m0.shape();

        (sh[0], sh[1])
    }
}

/// Peform a step of the Spin-Langevin stochastic differential equation (Stratonovich form)
/// using a 2nd order nonlinear Magnus propagator
///
///      \dd M   =  ( H(M) + \eta H(M) \cross M ) \cross M  \dd t + \sqrt(b) \dd xi(t) \cross M
///
/// Parameters:
/// work: SpinLangevinWorkpad, arrays of instances x spins x (3D x 4) SIMD packets
///         i.e. a total of (4*instances) x spins  3D Euclidean vectors
/// haml_update: Function pdate the local fields due to the spins at time t. Should read/modify
///             an ArrayView1<SpinVector3DAligned4xf64>, where the array dimension is over spin indices
/// eta : Dissipation strength
/// b: stochastic noise strength  (Should be proportional to $ K_b T \eta$ for a temperature T. See note)
/// rng: an RNG engine
/// rand_xi_f: Noise increment process. (Typically normalized Gaussian noise)
///
/// NOTE ON SEMICLASSICAL PHYSICS
///     The Spin-Langevin equation is obtained as the semiclassical limit of N unentangled
///     spin S particles interacting according to a quantum Hamiltonian on the spin states. It is
///     numerically optimized for N-body SO(3) dynamics due to the Lie algebra isomorphism
///         [ S_x,  S_y ] = i S_z         <--->          e_x \cross e_y = e_z   (and cyclic perms.)
///     where S_i are the angular momentum operators of the spins and e_i are 3D Euclidean unit vectors.
///
///     In particular, for the semiclassical limit of a N-spin-1/2 Hamiltonian in terms of Pauli matrices,
///     as $S_i = \frac{1}{2} \sigma_i  (\hbar \equiv 1)$,  each K-body interaction term should be
///     rescaled by 2^K. Additionally, the single-qubit coupling $\eta$ of the open system dynamics
///     should be rescaled by 2 in the Spin-Langevin equation. Failing to rescale will result
///     in (likely incorrect) dynamics over an incorrect time scale.
///
///     Nuclear/Particle physics applications should similarly rescale by the gyromagnetic ratio
///     where appropriate so that the Hamiltonian is in terms of S_i operators rather than
///     magnetic moments.
///
/// NOTE ON SDE FORM:
///     The SDE stepping method used here is based on the Stratonovich form.
///     However, the Ito and Stratonovich forms of the Spin-Langevin equations are the same
///     to accuracy $ O( \eta * k_b T * \delta_t )$. The Stratonovich form is preferred as it corresponds
///     to the physical limit where the correlation time of the noise source goes to zero.
///
///
/// Useful References:
/// 1.  Jayannavar, A. M. Brownian motion of spins; generalized spin Langevin equation.
///     Z. Physik B - Condensed Matter 82, 153–156 (1991).
/// 2.  Albash, T. & Lidar, D. A. Demonstration of a Scaling Advantage for a Quantum Annealer over
///     Simulated Annealing. Phys. Rev. X 8, 031016 (2018).
///
pub fn spin_langevin_step<Fh, R, Fr>(
    m0: &Array2<SpinVector3DAligned4xf64>, mf: &mut Array2<SpinVector3DAligned4xf64>,
    t0: f64, delta_t : f64,
    work :&mut SpinLangevinWorkpad,
    eta: f64, b: f64,
    haml_fn: Fh,
    rng: &mut R,
    rand_xi_f: Fr,
) where Fh: Fn(f64, &ArrayView1<SpinVector3DAligned4xf64>, &mut ArrayViewMut1<SpinVector3DAligned4xf64>),
        R: Rng + ?Sized,
        Fr: Fn(&mut R) -> SpinVector3DAligned4xf64
{
    let h_shape = work.shape();
    let t1 = t0 + delta_t/2.0;
    let t2 = t0 + delta_t;
    let delta_t = Aligned4xf64::from(delta_t);

    assert_eq!(m0.raw_dim(), work.h0.raw_dim());
    assert_eq!(mf.raw_dim(), m0.raw_dim());
    assert!(b >= 0.0, "Stochastic strength must be non-negative");

    let b_sqrt = Aligned4xf64::from(b.sqrt());
    // Populate random noise arrays
    let noise_1 = &mut work.chi1;
    let noise_2 = &mut work.chi2;
    for (chi1, chi2) in itertools::zip(noise_1.iter_mut(), noise_2.iter_mut()){
        *chi1 = rand_xi_f(rng) * b_sqrt;
        *chi2 = rand_xi_f(rng) * b_sqrt;
    }
    // Hamiltonian field update
    let h_update = |t: f64, h: &mut Array2<SpinVector3DAligned4xf64>, m: & Array2<SpinVector3DAligned4xf64> |{
        for (mut h_row, m_row) in
                h.axis_iter_mut(Axis(0)).zip(m.axis_iter(Axis(0)))
        {
            haml_fn(t, &m_row, &mut h_row);
            sl_add_dissipative(&mut h_row, & m_row, eta);
        }
    };
    // Spin propagation update
    let m_update = |omega: &Array2<SpinVector3DAligned4xf64>, spins_t0: &Array2<SpinVector3DAligned4xf64>,
                            spins_tf: &mut Array2<SpinVector3DAligned4xf64>|
    {
        let mut phi : Matrix3d4xf64 = Zero::zero();
        for (om, m0, mf) in itertools::multizip((omega.iter(), spins_t0.iter(), spins_tf.iter_mut(),)){
            cross_exponential_vector3d(om, &mut phi);
            phi.mul_to(m0, mf);
            let n_sq: Aligned4xf64 = mf.x*mf.x + mf.y*mf.y + mf.z*mf.z;
            let n = n_sq.map(f64::sqrt);
            *mf /= n;
        }
    };

    //let m0 = &work.m0;
    let haml_10 = &mut work.h0;
    let haml_11 = &mut work.h1;
    let haml_12 = &mut work.h2;

    // The nonlinear Magnus Expansion to 2nd order is as follows:
    //
    // STAGE 1
    // m_10  =  m_0,
    // H_{10} = H(t_0, m0),     H_{11} = H(t_1, m0)     H_{12} = H(t_2, m0)
    // \Omega_{11}  =  (\delta_t / 4) ( H_{10}  + H_{11} ) + b \sqrt{\delta_t/2} \chi_1
    // \Omega_{12} = (\delta_t / 6) (H_{10} + 4 H_{11} + H_{12} + b \sqrt{\delta_t/2} (\chi_1 + \chi_2)
    //
    // STAGE 2
    // m_{20} = m0,    m_{21} = \exp{\Omega_{11}} m_0,    m_{22} = \exp{\Omega_{12}} m_0
    // H_{20} =  H_{10},    H_{21} =H(t_1, m_{21}),     H_{22} = H(t_2, m_{22}
    // \Omega_2 = (\delta_t / 6) (H_{20} + 4 H_{21} + H_{22} + b \sqrt{\delta_t/2} (\chi_1 + \chi_2)
    //
    // Final propagation:
    // m[\delta_t] :=  \exp{\Omega_{22}} m_0

    // Stage 1 Computation
    h_update(t0, haml_10, m0);
    h_update(t1, haml_11, m0);
    h_update(t2, haml_12, m0);


    // Generator updates
    let omega_11 = &mut work.omega1;
    for (h0, h1, o1, chi1) in itertools::multizip((haml_10.iter(), haml_11.iter(), omega_11.iter_mut(), noise_1.iter())){
        *o1 = (h0 + h1) * Aligned4xf64::from(delta_t / 4.0)
            + chi1 * (delta_t / 2.0).map(f64::sqrt);
    }
    let omega_12 = &mut work.omega2;
    for (h0, h1, h2, o2, chi1, chi2) in itertools::multizip(
            (haml_10.iter(), haml_11.iter(), haml_12.iter(), omega_12.iter_mut(), noise_1.iter(), noise_2.iter()))
    {
        *o2 = (h0 + h1 * Aligned4xf64::from(4.0) + h2) * (delta_t / 6.0)
            + (chi1 + chi2) * (delta_t/2.0).map(f64::sqrt);
    }

    let spins_t0 = m0;
    let spins_t = mf;
    let haml_20 = haml_10;
    let haml_21 = haml_11;
    let haml_22 = haml_12;

    // Stage 2 computation

    // Evaluate m21 then update H21
    m_update(&*omega_11, spins_t0, spins_t);
    h_update(t1, haml_21, &*spins_t);

    // Evaluate m22 then update H22
    m_update(&*omega_12, spins_t0, spins_t);
    h_update(t2, haml_22, &*spins_t);


    // Finally evaluate \Omega_2
    let omega2 = &mut work.omega2;
    for (h0, h1, h2, o2, chi1, chi2) in itertools::multizip(
            (haml_20.iter(), haml_21.iter(), haml_22.iter(), omega2.iter_mut(), noise_1.iter(), noise_2.iter())){
        *o2 = (h0 + h1 * Aligned4xf64::from(4.0) + h2) * (delta_t / 6.0)
            + (chi1 + chi2) * (delta_t/2.0).map(f64::sqrt);
    }

    // Propagate m[0] to m[\delta_t]
    m_update(&*omega2, spins_t0, spins_t);

}

//pub fn spin_langevin_me2(
//    m0: ArrayView3<f64>,
//
//){
//
//}



//fn spin_langevin_dmdt(
//    hamiltonian_fields: &Array1<Aligned4xf64>,
//    spin_trajectories: &Array2<Aligned4xf64>,
//    dmdt_array: &mut Array2<Aligned4xf64>,
//    chi: f64
//)
//{
//    let h_shape = hamiltonian_fields.raw_dim();
//    let spins_shape = spin_trajectories.raw_dim();
//    let dm_shape = dmdt_array.raw_dim();
//    if h_shape[0] != spins_shape[0]{
//        panic!(format!("spin_langevin_dmdt: Dimension mismatch: h={:?} , m={:?} ", h_shape, spins_shape))
//    }
//
//    if dm_shape != spins_shape{
//        panic!(format!("spin_langevin_dmdt: Dimension mismatch: m={:?} , dm={:?} ", spins_shape, dm_shape))
//    }
//
//    for ((spins, mut dmdt),
//            h) in
//            spin_trajectories.axis_iter(Axis(0))
//                .zip(dmdt_array.axis_iter_mut(Axis(0)))
//                .zip(hamiltonian_fields.iter()){
//        for (si, dmi) in spins.iter().zip(dmdt.iter_mut()){
//            let mut cross_field = Aligned4xf64::default();
//            let mut triple_cross_field = Aligned4xf64::default();
//            cross_product_aligned_4xf64(h, si, &mut cross_field);
//            cross_product_aligned_4xf64(&cross_field, si, &mut triple_cross_field);
//
//            triple_cross_field *= chi;
//            *dmi = cross_field + triple_cross_field;
//            *dmi *= -1.0;
//        }
//    }
//}

#[cfg(test)]
mod tests{
    use ndarray::{Array1, Array2};
    use crate::util::simd_phys::vf64::{Aligned4xf64};
    use crate::semi::langevin::{spin_langevin_step, sl_add_dissipative, SpinVector3DAligned4xf64,
                                xyz_to_array_chunks};
    use num_traits::Zero;

    #[test]
    fn test_spin_langevin_dmdt(){
        let haml_arr = Array2::from_shape_vec((4,3),
           vec![ 1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.5, 0.5, 0.0,
                    0.5, -0.5, 0.0]).unwrap();
        let mut haml = Array1::from_elem((1,), Zero::zero());

        xyz_to_array_chunks(haml_arr.view(), haml.view_mut());
        let spins_arr = Array2::from_shape_vec((4, 3),
            vec![0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0,    0.0, 0.0, 1.0]
        ).unwrap();
        let mut spins = Array1::from_elem((1,), Zero::zero());
        xyz_to_array_chunks(spins_arr.view(), spins.view_mut());

        //let mut dm = Array1::from_elem((1,), ZERO_SPIN_ARRAY_3D);

        sl_add_dissipative(&mut haml.view_mut(), & spins.view(), 0.1);
    }
}
use alga::general::{RealField, ComplexField};
//use std::marker::PhantomData;
use integrators::gsl::{QAWC, QAGIU, QAGIL};
use itertools_num::linspace;
use integrators::Integrator;
use crate::util::{LinearInterpFn, InterpBounds};

pub trait Bath<N: RealField> {
    fn gamma(&self, omega: N) -> N;
    fn has_lamb_shift(&self) -> bool;
    fn lamb_shift(&self, omega:N) -> Option<N>;
}

pub struct OhmicBath<N: RealField>{
    eta: N,
    omega_c: N,
    beta: N,
    lamb: Option<LinearInterpFn<N>>
}

impl<N:RealField> OhmicBath<N>{
    pub fn new( eta:N, omega_c:N, beta:N) -> Self{
        Self{eta, omega_c, beta, lamb:None}
    }

    pub fn with_lamb_shift(self, ls_lo: N, ls_hi: N) -> Self{
        self.prepare_lamb_shift(ls_lo, ls_hi, 500)
    }

    fn prepare_lamb_shift(self, omega0_lo: N, omega0_hi:N, n :usize) -> Self{
        let mut lambs = Vec::<N>::new();
        lambs.resize(n, N::zero());

        let lo = omega0_lo.to_subset().unwrap();
        let hi = omega0_hi.to_subset().unwrap();
        assert!(hi > lo, "prepare_lamb_shift: upper limit must be greater than lower limit");

        let (c, d) = ((lo + hi)/2.0, (hi - lo)/2.0);
        // The integration limits should safely contain the desired range for omega0
        let ilo = c - 1.5*d;
        let ihi = c + 1.5*d;


        let mut qagil =QAGIL::new(100, ilo);
        let mut qagiu = QAGIU::new(100, ihi);
        let mut qawc = QAWC::new(100).with_range(ilo, ihi);
        let gamma_f64 = |x:f64| self.gamma(N::from_subset(&x)).to_subset().unwrap();

        for (w0,s) in linspace(lo, hi, n).zip(lambs.iter_mut()){
            //let mut qawc = QAWC::new(100).with_range(ilo, ihi);
            qawc = qawc.with_singularity(w0);

            let reswc = qawc.integrate(
                |x| -gamma_f64(x), 1.0e-8, 1.0e-8)
                .unwrap();
            let resil = qagil.integrate(|x| gamma_f64(x)/(w0 - x), 1.0e-8, 1.0e-8).unwrap();
            let resiu = qagiu.integrate(|x| gamma_f64(x)/(w0 - x), 1.0e-8, 1.0e-8).unwrap();

            let res = reswc.value
                + resil.value
                + resiu.value;
            *s = N::from_subset(&res);
        }

        let lambs_int = LinearInterpFn::new(lambs, omega0_lo, omega0_hi,
                                            InterpBounds::Zero);

        Self{lamb: Some(lambs_int), ..self}
    }
}


impl<N:RealField> Bath<N> for OhmicBath<N>{


    /// Calculates the ohmic rate of the frequency
    ///                2 pi * eta * w exp( - abs(w) / wc )
    ///         g =   ----------------------------------
    ///                     1 - np.exp(-self.beta*w)
    /// As w -> 0
    /// g = 2 pi eta /beta
    fn gamma(&self, omega:N) -> N {
        if ComplexField::abs(self.beta * omega) < N::from_subset(&1.0e-8){
            N::two_pi() * self.eta / self.beta
        } else {
            N::two_pi() * self.eta * omega * N::exp(- omega.abs()/self.omega_c)
                /(N::one() - N::exp(-self.beta*omega))
        }
    }

    fn has_lamb_shift(&self) -> bool {
        return self.lamb.is_some();
    }

    fn lamb_shift(&self, omega:N) -> Option<N>{
        self.lamb.as_ref().map(|lerp| lerp.at(omega))
    }
}

use complex_polygamma::trigamma;
use integrators::gsl::{QAGIL, QAGIU, QAWC, QAG};
use integrators::Integrator;
use itertools_num::linspace;
use num_traits::real::Real;
use num_traits::{One, Float};
use num_complex::Complex;

use crate::{ComplexField, RealField, RealScalar};
use crate::util::{InterpBounds, LinearInterpFn};

pub trait Bath<N: RealScalar> {
    fn gamma(&self, omega: N) -> N;
    fn has_lamb_shift(&self) -> bool;
    fn lamb_shift(&self, omega:N) -> Option<N>;
}

pub trait CorrelationFunction<N: RealScalar> {
    fn cor(&self, t: N) -> Complex<N>;

    /// Evaluate tau_sb, the decoherence time scale
    fn tau_sb(&self) -> Option<N>;
    /// Evaluate the bath correlation time and the coupling correlation time 
    /// Returns the tuple (tau_b, tau_sb)
    fn tau_b_sb(&self) -> Option<(N, N)>;
}

pub struct OhmicBath<N: RealScalar>{
    eta: N,
    omega_c: N,
    beta: N,
    lamb: Option<LinearInterpFn<N>>
}

impl<N:RealScalar> OhmicBath<N>{
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


impl<N:RealScalar> Bath<N> for OhmicBath<N>{


    /// Calculates the ohmic rate of the frequency
    ///                2 pi * eta * w exp( - abs(w) / wc )
    ///         g =   ----------------------------------
    ///                     1 - np.exp(-self.beta*w)
    /// As w -> 0
    /// g = 2 pi eta /beta
    fn gamma(&self, omega:N) -> N {

        if Real::abs(self.beta * omega) < N::from_subset(&1.0e-8){
            N::two_pi() * self.eta / self.beta
        } else {
            N::two_pi() * self.eta * omega * Real::exp(- Real::abs(omega)/self.omega_c)
                /(N::one() - Real::exp(-self.beta*omega))
        }
    }

    fn has_lamb_shift(&self) -> bool {
        return self.lamb.is_some();
    }

    fn lamb_shift(&self, omega:N) -> Option<N>{
        self.lamb.as_ref().map(|lerp| lerp.at(omega))
    }
}

impl<N:RealScalar+Float> CorrelationFunction<N> for OhmicBath<N>
where Complex<N> : ComplexField<RealField=N>
{
    fn cor(&self, t: N) -> Complex<N> {
        //let t = t.to_f64().unwrap();
        let z = Complex::<N>::new( Real::recip(self.omega_c * self.beta),
             t /self.beta,
        );
        let b = Complex::from(self.eta / (self.beta * self.beta) ) * (
            trigamma(z.conj() + One::one()).unwrap()
            + trigamma(z).unwrap()
        );

        return b;
    }

    /// Evaluate tau_sb, the decoherence time scale of the ohmic bath
    fn tau_sb(&self) -> Option<N>{
        let mut qagiu = QAGIU::new(100, 0.0);
        let resiu = qagiu.integrate(
            |t: f64| self.cor(N::from_f64(t).unwrap()).norm().to_f64().unwrap(),
            1.0e-8, 1.0e-8).unwrap();
        //println!("val={}, err={}",resiu.value, resiu.error);

        Some(N::from_f64(1.0/resiu.value).unwrap())
        
    }

    fn tau_b_sb(&self) -> Option<(N, N)> {
        // let mut qagiu = QAGIU::new(100, 0.0);
        // let resiu_b = qagiu.integrate(
        //     |t: f64| t * self.cor(N::from_f64(t).unwrap()).norm().to_f64().unwrap(),
        //  1.0e-8, 1.0e-8).unwrap();
        // let resiu_sb = qagiu.integrate(
        //     |t: f64| self.cor(N::from_f64(t).unwrap()).norm().to_f64().unwrap(),
        //  1.0e-8, 1.0e-8).unwrap();
        let tau_sb = self.tau_sb().unwrap();
        let tau_sb_f64 = tau_sb.to_f64().unwrap();
        let mut qag = QAG::new(100)
            .with_range(0.0, tau_sb_f64);
        let resqag = qag.integrate(
            |t: f64| t * self.cor(N::from_f64(t).unwrap()).norm().to_f64().unwrap(), 
            1.0e-8,1.0e-8).unwrap();
        //println!("val={}, err={}",resiu.value, resiu.error);

        Some((N::from_f64(resqag.value * tau_sb_f64).unwrap(), 
              tau_sb))
    }


    
}

#[cfg(test)]
mod tests{
    use core::f64::consts::PI;
    use super::{OhmicBath, Bath, CorrelationFunction};
    use integrators::Integrator;
    use integrators::gsl::QAGIU;
    use itertools_num::linspace;


    #[test]
    fn test_correlation(){
        let temp = 2.6;
        let bath = OhmicBath::new(1.0e-4, 8.0 * PI, 1.0/temp);

        for t in linspace(0.0, 10.0, 101){
            let ct = bath.cor(t);
            println!("{:3.2},  {}", t, ct.norm());
        }

        let (tau_b, tau_sb) = bath.tau_b_sb().unwrap();
        //let tau_sb = bath.tau_sb().unwrap();
        // let mut qagiu = QAGIU::new(100, 0.0);
        // let resiu = qagiu.integrate(|t| bath.cor(t).norm(), 1.0e-8, 1.0e-8).unwrap();
        println!("tau_b = {},\ntau_sb= {}", tau_b, tau_sb);
        
    }
}
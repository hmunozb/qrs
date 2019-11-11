//use interpolation::{lerp,Lerp};
use alga::general::{RealField};

pub enum InterpBounds<N: RealField>{
    Ends,
    Zero,
    Fill(N),
    Surround(N, N)
}

pub struct LinearInterpFn<N: RealField>{
    values: Vec<N>,
    bcs: InterpBounds<N>,
    lo: N,
    hi: N,
    dt: N
}

impl<N:RealField> LinearInterpFn<N>{
    pub fn new(values: Vec<N>, lo: N, hi:N, bcs: InterpBounds<N> ) -> Self{
        let n = values.len();
        assert_ne!(n, 0, "LinearInterpFn values must be non-empty.");
        assert!(hi > lo, "LinearInterpFn upper limit must be strictly greater than the lower limit.");
        let dt = (hi - lo)/ N::from_subset(&((n - 1) as f64));

        Self{values, bcs, lo, hi, dt}
    }

    pub fn at(&self, x: N) -> N{
        if x <= self.lo || x >= self.hi {
            match self.bcs {
                InterpBounds::Ends => {
                    return  if x<= self.lo {self.values.first().unwrap().clone() }
                            else { return self.values.last().unwrap().clone() };
                }
                InterpBounds::Zero => { return N::zero();}
                InterpBounds::Fill(f) => {return f;}
                InterpBounds::Surround(a, b) => {
                    return  if x<= self.lo { a }
                            else { b };
                }
            }
        }
         else {
            let k: N = (x - self.lo) /self.dt;
            let m = k.floor();
            let dxdt: N  = (x - m * self.dt)/self.dt;

            let m = m.to_subset().unwrap() as usize;
            if m==self.values.len() { return self.hi}

            let (y0, y1) = (self.values[m], self.values[m+1]);
            return y0*(N::one() - dxdt) + y1*dxdt;
            //return lerp(&self.values[m], &self.values[m+1], &x)
        }
    }
}

mod tests{

    #[test]
    fn test_linear_interp_fn(){
        use super::*;

        let li = LinearInterpFn::new(
            vec![0.0,1.0,4.0], 0.0, 2.0,InterpBounds::Ends);

        assert_eq!(li.at(-0.1), 0.0);
        assert_eq!(li.at(0.0), 0.0);
        assert_eq!(li.at(0.9), 0.9);
        assert_eq!(li.at(1.5), 2.5);
        assert_eq!(li.at(2.0), 4.0);

    }
}
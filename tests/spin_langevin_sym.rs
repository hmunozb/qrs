use std::fmt;
use std::fmt::Display;
use std::mem::swap;

use itertools::Itertools;
use nalgebra::Vector3;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use num_traits::{Float, Zero};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;
use simd_phys::r3::Vector3d4xf64;
use simd_phys::vf64::Aligned4xf64;

use qrs::semi::langevin::{spin_langevin_step,
                          SpinLangevinWorkpad, StepResult, xyz_to_array_chunks};

pub struct SVDStats{
    i: u64,
    t: f64,
    q: i64,

    mean: Vector3<f64>,
    mean_sq: Vector3<f64>,
    mean_4: Vector3<f64>
}


impl Display for SVDStats{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{:07}, {:>9.1}, {:^04}, ", self.i, self.t, self.q)?;
        write!(f, "{:+15.14}, {:+15.14}, {:+15.14}, ", self.mean.x, self.mean.y, self.mean.z)?;
        write!(f, "{:7.6}, {:7.6}, {:7.6}, ", self.mean_sq.x, self.mean_sq.y, self.mean_sq.z)?;
        write!(f, "{:7.6}, {:7.6}, {:7.6}, ", self.mean_4.x, self.mean_4.y, self.mean_4.z)
    }
}

#[test]
fn test_sym_langevin_step(){

    let tf = 20_000.0;
    let delta_t = 0.1;
    let b = 0.0 * 1.0e-4;
    let eta = 1.0e-4;
    let num_traj = 100;
    let time_points = itertools_num::linspace(0.0, tf, 21).collect_vec();

    let mut rngt = thread_rng();
    let mut seed_seq = [0u8; 32];
    rngt.fill_bytes(&mut seed_seq);
    let mut rng =  Xoshiro256Plus::from_seed(seed_seq);

    let sched = |t: f64| {t /tf};
    //let anneal_sched_a = | s: f64| {(0.97 - s).min(1.0)};
    //let anneal_sched_b = | s: f64| {0.03 + 0.97*s};

    let sched_a = |t:f64|  sched_a_dw2kq_nsi(sched(t));
    let sched_b = |t:f64|  sched_b_dw2kq_nsi(sched(t));

    //let sched_a = |t: f64| 0.0;
    //let sched_b = |t: f64| 10.0;

    let haml_fn = |t: f64, m: &ArrayView1<Vector3d4xf64>, h: &mut ArrayViewMut1<Vector3d4xf64>|{
        for (i, hi) in (0usize..3).zip(h.iter_mut())
        {
            let mut hz :Aligned4xf64 = Zero::zero();
            if i == 0 {hz += -1.0;}
            unsafe{
                //hz += -( m.uget((i+1)%3)[2] + m.uget((i+2)%3)[2] )
                hz += m.uget((i+1)%3)[2] * -1.0;
                hz += m.uget((i+2)%3)[2] * -1.0;
            }
            let hx = Aligned4xf64::from(-sched_a(t));
            hz *= sched_b(t);

            hi[0] = hx;
            hi[1] = Aligned4xf64::zero();
            hi[2] = hz;
        }
    };

    let rand_fn = |rng: &mut Xoshiro256Plus|
                   -> Vector3d4xf64{
        let mut sv : Vector3d4xf64 = Zero::zero();

        sv[0] = if b > 0.0 {rng.sample(StandardNormal)} else { Zero::zero() } ;
        sv[1] = if b > 0.0 {rng.sample(StandardNormal)} else { Zero::zero() } ;
        sv[2] = if b > 0.0 {rng.sample(StandardNormal)} else { Zero::zero() } ;

        sv
    };

    let mut measure_stats = |i:u64, t: f64, m: &Array2<Vector3d4xf64>|
                             -> Result<(), ()>
        {
            for (q, cols) in m.gencolumns().into_iter().enumerate(){
                let (r, r_sq, r_4, _n) = sv3x4f64_stats(cols);

                let svd_stats = SVDStats{i, t, q: q as i64, mean: r, mean_sq: r_sq, mean_4: r_4};
                println!( "{}", svd_stats);
            }

//            for (row, pi) in m.genrows().into_iter().zip(overlaps.iter_mut()){
//                *pi = tgt_overlap(row.view(), svd_target.view());
//            }
//            let n = num_samples as f64;
//            let pm : f64 = overlaps.iter().map(|x| {x.sum_reduce()}).sum::<f64>()/n;
//            let pm2 : f64 = overlaps.iter().map(|x| {x.sum_sq_reduce()}).sum::<f64>()/n;
//            let pm4 : f64 = overlaps.iter().map(|&x| {(x*x).sum_sq_reduce()}).sum::<f64>()/n;
//            let svd_stats = SVDStats{i, t, q: -1, mean: Vector3::new(0.0, 0.0, pm),
//                mean_sq: Vector3::new(0.0, 0.0, pm2),
//                mean_4: Vector3::new(0.0, 0.0, pm4)};
//            println!( "{}", svd_stats);
            // Evaluate the total magnetization
            //let mag= m.sum_axis(Axis(0));

            Ok(())
        };

    let arr_shape = (num_traj, 3);
    let mut m0 : Array2<Vector3d4xf64> = Array2::from_elem(arr_shape,Zero::zero());
    let mut mf = m0.clone();
    let mut work = SpinLangevinWorkpad::from_shape(arr_shape.0, arr_shape.1);

    for sv in m0.iter_mut(){
        sv[0] = Aligned4xf64::from(1.0);
    }

    let max_dt = delta_t * 10.0;
    let min_dt = delta_t * 1.0e-3;
    let mut dt = delta_t;

    let mut ti = 0;
    let mut i = 0;
    let mut t = 0.0;
    let mut ri = 0;

    loop{
        let next_t = t + dt;
        let res = spin_langevin_step(&m0, &mut mf, t, dt, &mut work, eta, b,
                           haml_fn, &mut rng, rand_fn);

        match res{
            StepResult::Accept(h)  => {
                dt = ((0.9 / h) * dt).min(max_dt);
                ri = 0;
            },
            StepResult::Reject(h)  =>{
                println!("Step {} (t = {}) rejected with h = {}", ti, t, h);
                let new_dt = dt / (1.1*h);
                if (new_dt < min_dt) && ri >= 3{
                    panic!("spin_langevin: Step size too small (dt = {})", new_dt);
                }
                dt = new_dt.max(min_dt);
                ri += 1;
                continue;
            }
        }

        if t >= time_points[i]{
            println!("t = {} ", t);
            measure_stats(i as u64, t, &m0);
            measure_stats(100 + i as u64, t, &work.h0);

            i += 1;
        }
        if t >= tf{
            break;
        }
        t = next_t;
        ti += 1;
        swap(&mut m0, &mut mf); // Set m0 to the current data
    }
}


/// Evaluate the average spin directions (x, y, z), reducing over the array view
/// and SIMD channel
fn sv3x4f64_stats(m: ArrayView1<Vector3d4xf64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, f64){
    let mut mean: Vector3<f64> = Zero::zero();
    let mut mean_sq: Vector3<f64> = Zero::zero();
    let mut mean_4: Vector3<f64> = Zero::zero();
    let mut mean_norm = 0.0;
    for mi in m.iter(){
        //reduce the SIMD
        let mi_sq = Vector3d4xf64::map(mi, |x| x*x);
        let mi_4 = Vector3d4xf64::map(&mi_sq, |x|x*x);
        let n = (mi_sq.x + mi_sq.y + mi_sq.z).sum_reduce();
        mean_norm += n;

        let r = Vector3d4xf64::map(mi, |x| x.sum_reduce());
        let r_sq = mi_sq.map( |x| x.sum_reduce());
        let r_4 = mi_4.map(|x| x.sum_reduce());
        mean += r;
        mean_sq += r_sq;
        mean_4 += r_4;
    }
    mean /= 4.0*m.len() as f64;
    mean_sq /= 4.0*m.len() as f64;
    mean_norm /= 4.0*m.len() as f64;
    mean_4 /= 4.0 * m.len() as f64;

    (mean, mean_sq, mean_4, mean_norm)
}

fn sched_a_dw2kq_nsi(s: f64) -> f64{
    let is = 1.0 - s;
    let is7 = f64::powi(is, 7);

    2.0 * f64::exp(3.17326 * s) * is7 * (
        207.253 * is * is  - 380.659  * is + 203.843)
}

fn sched_b_dw2kq_nsi(s: f64) -> f64{
    2.0 * (0.341734 + 6.71328*s + 32.9702*s*s)
}


//fn main(){
//    test_sym_langevin_step()
//}
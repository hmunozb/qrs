use std::iter::FromIterator;

use alga::general::{ComplexField, RealField};
use blas_traits::BlasScalar;
use num_traits::{One, Zero};
use num_complex::Complex;
use nalgebra::{DMatrix, DVector};
use itertools_num::linspace;
use log::{info, warn};
use crate::base::dense::*;
use crate::util::time_dep_op::TimeDepMatrix;
use num_traits::Float;
use crate::util::{EigJob, EigRangeData, EigResolver, change_basis};

pub struct TimePartitionOptions {
    pub nudge_final_partition: bool
}

/// This struct implements a Time-Partitioned Hamiltonian
///
pub struct TimePartHaml<'a, R: RealField>
    where Complex<R>: ComplexField<RealField=R> + BlasScalar
{
    basis_size: u32,
    time_range: (R, R),
    time_partitions: Vec<R>,
    partition_midpoints: Vec<R>,
    basis_sequence: Vec<Op<R>>,
    projector_sequence: Vec<Op<R>>,
    haml: TimeDepMatrix<'a, Complex<R>>,
    haml_sequence: Vec<TimeDepMatrix<'a, Complex<R>>>,
}

impl<'a, R: RealField + Float> TimePartHaml<'a, R>
    where Complex<R>: ComplexField<RealField=R> + BlasScalar {
    pub fn new(haml: TimeDepMatrix<'a, Complex<R>>, basis_size: u32, t0: R, tf: R, num_partitions: usize) -> Self {
        let time_partitions = Vec::from_iter(
            linspace(t0, tf, num_partitions + 1));
        Self::new_with_partitions(haml, basis_size, time_partitions)
    }

    pub fn new_with_partitions(haml: TimeDepMatrix<'a, Complex<R>>, basis_size: u32,
                               time_partitions: Vec<R>) -> Self {
        if time_partitions.len() == 0 {
            panic!("TimePartitionedHamiltonian time_partitions must be non-empty.");
        }
        for (t, t_next) in time_partitions.iter().zip(time_partitions.iter().skip(1)) {
            if t >= t_next {
                panic!("TimePartitionedHamiltonian time_partitions must be sorted and increasing.");
            }
        }

        let t0 = time_partitions.first().unwrap().clone();
        let tf = time_partitions.first().unwrap().clone();

        let time_range = (t0.clone(), tf.clone());
        let partition_midpoints: Vec<R> =
            Vec::from_iter(time_partitions.iter()
                .skip(1).zip(time_partitions.iter())
                .map(|(a1, a0)| (a0.clone() + a1.clone()) / R::from_f64(2.0).unwrap()));
        let basis_sequence = Vec::new();
        let projector_sequence = Vec::new();
        let haml_sequence = Vec::new();
        let mut me = Self {
            basis_size,
            time_range,
            time_partitions,
            partition_midpoints,
            basis_sequence,
            projector_sequence,
            haml,
            haml_sequence,
        };
        me.construct_sequence();
        me
    }

    pub fn time_range(&self) -> (R, R){
        self.time_range.clone()
    }

    pub fn partitions(&self) -> &[R] {
        &self.time_partitions
    }

    /// Construct the truncated Hamiltonian sequence over the midpoints of each partition
    fn construct_sequence(&mut self) {
        let n = self.haml.shape().0;
        let mut eiger: EigResolver<Complex<R>> = EigResolver::new_eiger(
            n as u32, EigJob::ValsVecs,
            EigRangeData::num_range(0, self.basis_size as i32));

        for t in self.partition_midpoints.iter() {
            let ht = eiger.borrow_matrix();
            self.haml.eval_to(ht, *t);
            eiger.eig();

            self.basis_sequence.push(eiger.vecs().clone());
            let h_local = self.haml.map(
                |m| change_basis(m, eiger.vecs()));
            self.haml_sequence.push(h_local);
        }

        for p in 0..(self.partition_midpoints.len()-1){
            let mut wi = self.basis_sequence[p].clone();

            let wj = &self.basis_sequence[p+1];
            let proj = wi.ad_mul( wj);
            self.projector_sequence.push(proj);
        }

    }

    pub fn auto_refine(self, strength: usize, tol: f64, max_rounds: u32) -> Self{
        let mut me = self;
        let mut previous_parts = me.time_partitions.len();
        for _ in 0..max_rounds{
            me = me.refine_partition(strength, tol);
            if me.time_partitions.len() == previous_parts{
                return me;
            }
        }
        warn!("Partition Refinement failed to converge after {} rounds. Stopping.", max_rounds);
        return me;
    }

    ///Estimates the trace loss incurred through each partition by projecting a matrix
    /// in the truncated Hamiltonian basis from start of the partition onto the basis at
    /// the middle, and then onto the end of the partition.
    /// Overall, the lower the trace loss at a strength number, the more faithful the truncated
    /// Hamiltonian sequence is at that number of lowest energy levels
    fn refine_partition(self, strength: usize, tol: f64) -> Self {
        let num_partitions = self.partition_midpoints.len();
        let n = self.haml.shape().0;
        let id_diagonal: DVector<Complex<R>> = DVector::from_fn(
            n, |i, _| if i < strength { Complex::one() } else { Complex::zero() });
        let id0: DMatrix<Complex<R>> = DMatrix::from_diagonal(&id_diagonal);
        let mut eiger: EigResolver<Complex<R>> = EigResolver::new_eiger(
            n as u32, EigJob::ValsVecs,
            EigRangeData::num_range(0, self.basis_size as i32));

        let mut get_eigs = |t: &R| {
            self.haml.eval_to(eiger.borrow_matrix(), t.clone());
            eiger.eig();
            eiger.vecs().clone()
        };

        // Initial Conditions for the first partition
        // Reupdated for each partition in the loop
        let mut t0 = self.time_partitions[0].clone();
        let mut new_time_intervals: Vec<R> = Vec::new();
        let mut vecs_init = get_eigs(&t0);

        new_time_intervals.push(t0.clone());

        for (i, (t1, tmid)) in self.time_partitions.iter().skip(1)
            .zip(self.partition_midpoints.iter())
            .enumerate() {
            let vecs_mid = get_eigs(tmid);
            let vecs_end = get_eigs(t1);
            let projector = (vecs_init.ad_mul(&vecs_mid)) * (vecs_mid.ad_mul(&vecs_end));
            let id1 = change_basis(&id0, &projector);
            let trace_loss = (id1.trace() as Complex<R>).re.to_f64().unwrap();
            if trace_loss > tol {
                let order = (1.0 + (trace_loss.abs() / tol).log2()).round() as usize;
                info!("Partition {}/{} will be refined: Tr Loss={}, r={}",
                      i, num_partitions, trace_loss, order);
                for new_t in linspace(t0.clone(), t1.clone(), 1 + order) {
                    new_time_intervals.push(new_t);
                }
            }
            // Reinit for next loop
            t0 = t1.clone();
            vecs_init = vecs_end;
        }
        // Append final point
        new_time_intervals.push(self.time_partitions.last().unwrap().clone());
        Self::new_with_partitions(self.haml, self.basis_size, new_time_intervals)
    }

    fn partition_of(&self, t: R) -> Option<usize>{
        for (i, ti)  in self.time_partitions.iter().enumerate(){
            if t >= *ti{
                return Some(i);
            }
        }
        None
    }

    fn p_eiger(&self) -> EigResolver<Complex<R>>{
        let k = self.basis_size;
        let eiger: EigResolver<Complex<R>> = EigResolver::new_eiger(
            k , EigJob::ValsVecs, EigRangeData::all());
        eiger
    }

    pub fn basis_size(&self) -> u32{
        return self.basis_size;
    }
    pub fn eig_p(&self, t: R, p : Option<usize>) -> (DVector<R>, DMatrix<Complex<R>>){
        let part = match p {
            None => match self.partition_of(t.clone())
                { None => panic!("The time {} is not within range", t), Some(tp) => tp },
            Some(part) => part
        };
        let mut eiger = self.p_eiger();
        self.haml_sequence[part].eval_to(eiger.borrow_matrix(), t);

        eiger.into_eigs()
    }

    pub fn transform_to_partition(&self, op: &Op<R>, p: usize) -> Op<R>{
        let vecs = &self.basis_sequence[p];
        change_basis(op,  vecs)
    }

    pub fn advance_partition(&self, op: &Op<R>, p0: usize ) -> Op<R>{
        let projs = &self.projector_sequence[p0];
        change_basis(op, projs)
    }
}


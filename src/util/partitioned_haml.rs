use std::iter::FromIterator;

use itertools_num::linspace;
use log::{info, trace, warn, log_enabled, debug};
use nalgebra::{DMatrix, DVector, Dynamic, U1};
use ndarray::Array1;
use num_complex::Complex;
//use blas_traits::{BlasScalar, Tsyheevx};
use num_traits::{One, ToPrimitive, Zero};
use num_traits::Float;

use qrs_core::eig::{EigJob, EigQRep, EigRange};
use qrs_core::eig::dmatrix::EigScalar;
//use qrs_core::reps::dense::{DenseQRep, Op};
use qrs_core::eig::QEiger;
use qrs_core::reps::matrix::{DenseQRep, Op};

//use crate::util::degen::{handle_degeneracies_vals, handle_phases, degeneracy_detect, handle_degeneracies};
use crate::{ComplexScalar, RealScalar};
use crate::util::{EigResolver, TimeDependentOperator};
//use qrs_core::util::array::{change_basis};
use crate::util::change_basis;
use crate::util::time_dep_op::TimeDepMatrix;

// pub type TimeDepQOp<'a, R: RealScalar> = TimeDependentOperator<'a, R, Complex<R>,
//     DenseQRep<Complex<R>>, Op<Complex<R>> >;

pub type TimeDepQOp<'a, R> = TimeDepMatrix<'a, Complex<R>>;

pub struct TimePartitionOptions {
    pub nudge_final_partition: bool
}

/// This struct implements a Time-Partitioned Hamiltonian
///
pub struct TimePartHaml<'a, R: RealScalar>
    where Complex<R>: ComplexScalar<R=R>
    //where Complex<R>: ComplexField<RealField=R> //+ BlasScalar
{
    basis_size: u32,
    time_range: (R, R),
    time_partitions: Vec<R>,
    partition_midpoints: Vec<R>,
    basis_sequence: Vec<Op<Complex<R>>>,
    projector_sequence: Vec<Op<Complex<R>>>,
    haml: TimeDepQOp<'a, R>,
    haml_sequence: Vec<TimeDepQOp<'a, R>>,
}

impl<'a, R> TimePartHaml<'a, R>
where R: RealScalar + Float,  Complex<R>: EigScalar<R=R>
      //Complex<R>: ComplexField<RealField=R> //+ Tsyheevx
//where Complex<R> : ComplexScalar<R>
//where R: RealField
    //where Complex<R>: ComplexField //<RealField=R> //+ BlasScalar
{
    pub fn new(haml: TimeDepQOp<'a, R>, basis_size: u32, t0: R, tf: R, num_partitions: usize) -> Self {
        let time_partitions = Vec::from_iter(
            linspace(t0, tf, num_partitions + 1));
        Self::new_with_partitions(haml, basis_size, time_partitions)
    }

    pub fn new_with_partitions(haml: TimeDepQOp<'a, R>, basis_size: u32,
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
        let tf = time_partitions.last().unwrap().clone();

        let time_range = (t0.clone(), tf.clone());
        let mut partition_midpoints: Vec<R> =
            Vec::from_iter(time_partitions.iter()
                .skip(1).zip(time_partitions.iter())
                .map(|(a1, a0)| (a0.clone() + a1.clone()) / R::from_f64(2.0).unwrap()));
        // Nudge the final representative time to the end
        *partition_midpoints.last_mut().unwrap() = tf;
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

        let mut eiger: EigResolver<Complex<R>> =  QEiger::<Complex<R>, DenseQRep<Complex<R>>>
            ::make_eiger((n, n), EigJob::ValsVecs,
                     EigRange::IdxRange(0, self.basis_size as i32));

        for &t in self.partition_midpoints.iter() {
            let (_vals, vecs) = QEiger::<Complex<R>, DenseQRep<Complex<R>>>::eigh(&mut eiger,
                                                                                  &self.haml.eval(t));
            // let ht = eiger.borrow_matrix();
            // self.haml.eval_to(ht, *t);
            // eiger.eig();
            //let vecs = eiger.vecs().clone();
//            let vals = eiger.vals().as_slice().get(0..self.basis_size as usize).unwrap();
//            let vals = DVector::from_column_slice(vals);
//            let degens = degeneracy_detect(&vals, None);
//            if degens.len() > 0{
//                info!("A degeneracy in the basis at time {} was handled.", t);
//                handle_degeneracies(&degens, &mut vecs);
//            }

            let h_local = self.haml.map(
                |m| change_basis(m, &vecs));

            self.basis_sequence.push(vecs);
            self.haml_sequence.push(h_local);
        }

        for p in 0..(self.partition_midpoints.len()-1){
            let wi = self.basis_sequence[p].clone();

            let wj = &self.basis_sequence[p+1];
            let proj = wi.ad_mul( wj);
            if log_enabled!(log::Level::Debug){
                debug!("proj {} tr = {}", p, proj.trace())
            }
            self.projector_sequence.push(proj);
        }

    }

    pub fn auto_refine(self, strength: usize, tol: f64, max_rounds: u32) -> Self{
        let mut me = self;
        let mut previous_parts = me.time_partitions.len();

        for _ in 0..max_rounds{
            me = me.refine_partition(strength, tol);
            let current_parts = me.time_partitions.len();
            if current_parts == previous_parts{
                return me;
            }
            previous_parts = current_parts;
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
            self.basis_size as usize, |i, _| if i < strength { Complex::one() } else { Complex::zero() });
        let id0: DMatrix<Complex<R>> = DMatrix::from_diagonal(&id_diagonal);
        let mut eiger: EigResolver<Complex<R>> =  QEiger::<Complex<R>, DenseQRep<Complex<R>>>
                ::make_eiger((n, n), EigJob::ValsVecs, EigRange::IdxRange(0, self.basis_size as i32));
            // EigResolver::new_eiger(
            // n as u32, EigJob::ValsVecs,
            // EigRangeData::idx_range(0, self.basis_size as i32));

        let mut get_eigs = |t: &R| -> Op<Complex<R>>{
            let ht = self.haml.eval(*t);
            let (_vals, vecs) = QEiger::<Complex<R>, DenseQRep<Complex<R>>>::eigh(&mut eiger,&ht);
            //self.haml.eval_to(eiger.borrow_matrix(), t.clone());
            //eiger.eig();
            //eiger.vecs().clone()
            vecs
        };

        // Initial Conditions for the first partition
        // Reupdated for each partition in the loop
        let mut t0 = self.time_partitions[0].clone();
        let mut new_time_intervals: Vec<R> = Vec::new();
        let mut vecs_init = get_eigs(&t0);

        for (i, (t1, tmid)) in self.time_partitions.iter().skip(1)
            .zip(self.partition_midpoints.iter())
            .enumerate() {
            let vecs_mid = get_eigs(tmid);
            let vecs_end = get_eigs(t1);
            let projector = (vecs_init.ad_mul(&vecs_mid)) * (vecs_mid.ad_mul(&vecs_end));
            let id1 = change_basis(&id0, &projector);

            let trace_loss = 1.0 - (id1.trace() as Complex<R>).re.to_f64().unwrap()/(strength as f64);
            new_time_intervals.push(t0.clone());
            trace!("Partition {}/{} trace loss: {}",i+1, num_partitions, trace_loss);
            if trace_loss > tol {
                let order = (1.0 + (trace_loss.abs() / tol).log2()).round() as usize;
                info!("***Partition {}/{} will be refined: Tr Loss={}, r={}",
                      i, num_partitions, trace_loss, order);
                //linspace is endpoint inclusive
                //we only need to append the intermediate refinement points
                for new_t in linspace(t0.clone(), t1.clone(), 2 + order)
                                    .take(order + 1).skip(1) {
                    new_time_intervals.push(new_t);
                }
            }
            // Reinit for next loop
            t0 = t1.clone();
            vecs_init = vecs_end;
        }
        // Append final point
        new_time_intervals.push(self.time_partitions.last().unwrap().clone());
        if new_time_intervals.len() > self.time_partitions.len()
        {
            Self::new_with_partitions(self.haml, self.basis_size, new_time_intervals) }
        else{
            self
        }
    }

    fn partition_of(&self, t: R) -> Option<usize>{
        for (i, ti)  in self.time_partitions.iter().enumerate(){
            if t >= *ti{
                return Some(i);
            }
        }
        None
    }

    // fn p_eiger(&self) -> EigResolver<Complex<R>>{
    //     let k = self.basis_size;
    //     let eiger: EigResolver<Complex<R>> = EigResolver::new_eiger(
    //         k , EigJob::ValsVecs, EigRangeData::all());
    //     eiger
    // }

    pub fn basis_size(&self) -> u32{
        return self.basis_size;
    }
    pub fn eig_p(&self, t: R, p : Option<usize>) -> (DVector<R>, Op<Complex<R>>){
        let part = match p {
            None => match self.partition_of(t.clone())
                { None => panic!("The time {} is not within range", t), Some(tp) => tp },
            Some(part) => part
        };
        //let mut eiger = self.p_eiger();

        let ht= self.haml_sequence[part].eval(t);
        let (vals, vecs) = DenseQRep::eig(&ht);
        //let (vals, vecs) = EigResolver::<Complex<R>>::eigh(,&ht);
        let vals : DVector<R> = DVector::from_vec(vals);
        //self.haml_sequence[part].eval_to(eiger.borrow_matrix(), t);

        //let (vals, vecs) = eiger.into_eigs();

//        if handle_degens{
//            handle_degeneracies_vals(&vals, &mut vecs, None);
//        }
//        if phases_gauge{
//            handle_phases(&mut vecs);
//        }

        (vals, vecs)
    }

    // pub fn canonical_basis_amplitudes(&self, i: usize, p: usize)
    //         -> nalgebra::MatrixSliceMN<Complex<R>, U1, Dynamic, U1, Dynamic> {
    //     let row = self.basis_sequence[p].row(i);
    //     row
    // }
    /// Gets the rows of the basis matrix from each index in cb.
    /// These are simply a subset of the amplitudes in the canonical basis
    pub fn sparse_canonical_basis_amplitudes(&self, cb: &[usize], p: usize) -> Op<Complex<R>>{
        let n = cb.len();
        let k = self.basis_size as usize;
        let e_basis = &self.basis_sequence[p];
        let mut cb_proj :Op<Complex<R>> = Op::zeros(n, k);
        for i in 0..n{
            let mut row = cb_proj.row_mut(i);
            row.copy_from(&e_basis.row(cb[i]));
        }

        cb_proj
    }
    /// For each basis matrix in the basis sequence, gets the rows of the basis
    /// matrix from each index in cb.
    /// These are simply a subset of the amplitudes in the canonical basis
    pub fn sparse_canonical_basis_projs(&self, cb: &[usize]) -> Vec<Op<Complex<R>>>{
        let mut cb_vec = Vec::new();
        for p in 0..self.basis_sequence.len(){
            cb_vec.push(self.sparse_canonical_basis_amplitudes(cb, p));
        }
        cb_vec
    }

    pub fn transform_to_partition(&self, op: &Op<Complex<R>>, p: usize) -> Op<Complex<R>>{
        let vecs = &self.basis_sequence[p];
        change_basis(op,  vecs)
    }

    pub fn advance_partition(&self, op: &Op<Complex<R>>, p0: usize ) -> Op<Complex<R>>{
        let projs = &self.projector_sequence[p0];
        change_basis(op, projs)
    }

    pub fn projector(&self, p0: usize) -> &Op<Complex<R>>{
        &self.projector_sequence[p0]
    }
}

#[cfg(test)]
mod tests{
    use cblas::c32;
    use num_complex::Complex;
    use num_complex::Complex64 as c64;

    use crate::base::pauli::matrix as pauli;
    use crate::util::{TimeDepMatrix, TimeDepMatrixTerm};

    use super::TimePartHaml;

    #[test]
    fn test_part_haml(){
        let tf = 10.0;
        let sx = pauli::sx::<f64>();
        let sz = pauli::sz::<f64>();

        let fx = |t: f64| Complex::from(10.0*(0.5 - (t/tf)));
        let fz = |t: f64| Complex::from(0.5);

        let hx = TimeDepMatrixTerm::new(&sx, &fx);
        let hz = TimeDepMatrixTerm::new(&sz, &fz);

        let haml_a = TimeDepMatrix{terms: vec![hx.clone()]};
        let haml_b = TimeDepMatrix{terms: vec![hz.clone()]};
        let haml_sum = TimeDepMatrix{terms: vec![hx, hz]};

        let haml = TimePartHaml::new(haml_sum, 2,0.0, tf, 10);
    }
}
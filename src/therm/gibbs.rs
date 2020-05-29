use ndarray::Array1;
use ndarray_stats::QuantileExt;
use num_traits::real::Real;
use qrs_core::ComplexScalar;
use qrs_core::quantum::{QRep, FDimQRep, QOp};
use qrs_core::eig::{QEiger, EigVecResult};
use qrs_core::util::iter::Sum;


pub struct GibbsResult<R, QOpT>{
    pub boltz_weights: Array1<R>,
    pub partition_z: R,
    pub evecs: QOpT
}

pub fn gibbs_state<'a, N: ComplexScalar, Q: QRep<N, OpRep=QOpT>, QEig: QEiger<N, Q>,
                QOpT: 'a + QOp<N, Rep=Q>, I: Iterator<Item=&'a QOpT>>(
    q_rep: Q,
    q_eiger: &mut QEig,
    haml: &QOpT,
    beta: N::R
) -> GibbsResult<N::R, QOpT>
where N::R : Real
{
    let (vals, vecs) = q_eiger.eigh(haml);
    let n = vals.len();
    let mut vals : Array1<N::R> = Array1::from(vals);
    let &e0 = vals.min().unwrap();
    for e in vals.iter_mut(){
        *e -= e0;
    }
    vals.mapv_inplace(|x| Real::exp(-beta * x ));
    let z_partfn = vals.sum();

    GibbsResult{boltz_weights: vals, partition_z: z_partfn, evecs: vecs.into_op()}

}

#[cfg(test)]
mod test{
    #[test]
    fn test_gibbs_state(){

    }
}
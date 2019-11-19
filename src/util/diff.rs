use crate::base::dense::*;
use alga::general::ComplexField;
use nalgebra::{DMatrix};
use smallvec::SmallVec;
use vec_ode::LinearCombination;

static GL_D_STENC_1 : [f64; 4] =[
    -2.7320508075688772935, 1.7320508075688772935,
    1.7320508075688772935, -0.73205080756887729353];

static GL_D_STENC_2 : [f64; 4] = [
    0.73205080756887729353, -1.7320508075688772935,
    -1.7320508075688772935, 2.7320508075688772935
];

pub fn four_point_gl<S: ComplexField>(vecs: &[DMatrix<S>],
                                      d1: &mut  DMatrix<S>, d2: &mut DMatrix<S>){
    let c1 : SmallVec<[S;4]> = GL_D_STENC_1.iter()
                                            .map(|k| S::from_subset(k))
                                            .collect();
    let c2 : SmallVec<[S;4]> = GL_D_STENC_2.iter()
                                            .map(|k| S::from_subset(k))
                                            .collect();
    d1.linear_combination(vecs, &c1);
    d2.linear_combination(vecs, &c2);
}
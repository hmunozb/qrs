use qrs_core::ComplexScalar;
use qrs_core::reps::matrix::{DenseQRep, Op};
pub use qrs_timed::timed_op::*;

pub type TimeDepMatrixTerm<'a, N> = TimeDependentOperatorTerm<'a, <N as ComplexScalar>::R, N, DenseQRep<N>, Op<N>>;
pub type TimeDepMatrix<'a, N> = TimeDependentOperator<'a, <N as ComplexScalar>::R, N, DenseQRep<N>, Op<N>>;


mod tests{
    use crate::ComplexField;
    #[test]
    fn test_time_dep_op(){
        use nalgebra::DMatrix;
        use crate::util::time_dep_op::{TimeDepMatrixTerm, TimeDepMatrix};
        use alga::general::ComplexField;
        use num_complex::Complex64 as c64;
        //use crate::base::DenseQRep;
        use crate::base::quantum::{TensorProd};

        let _1c = c64::new(1.0, 0.0);
        let _0c = c64::new(0.0, 0.0);
        let _i = c64::i();
        let _sx = [ _0c, _1c,
                    _1c, _0c];
        let _sz = [ _1c, _0c,
                    _0c, -_1c];

        let sx = DMatrix::from_row_slice(2, 2, &_sx);
        let sz = DMatrix::from_row_slice(2, 2, &_sz);
        let sx2: DMatrix<c64> = TensorProd::tensor_ref(&sx, &sx);
        let sz2: DMatrix<c64> = TensorProd::tensor_ref(&sz, &sz);
        let fx = |s:f64| c64::from_real(1.0 - s);
        let fz =|s: f64| c64::from_real(s);
        let hx = TimeDepMatrixTerm::new(&sx2, &fx);
        let hz = TimeDepMatrixTerm::new(&sz2, &fz);
        let haml = TimeDepMatrix{terms: vec![hx, hz]};

        let h1 = haml.eval(0.5);
        println!("t=0.5, h=\n{}", h1);

    }
}
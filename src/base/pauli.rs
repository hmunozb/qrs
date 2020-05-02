
pub mod dense{
    use std::iter::FromIterator;

    use crate::{ComplexField, ComplexScalar, RealField, RealScalar};
    use num_traits::{Zero, One};
    use num_complex::Complex;
    use crate::base::quantum::*;
    use qrs_core::reps::dense::*;
    use ndarray_linalg::Norm;
    //use nalgebra::DMatrix;


    pub enum Eigs{
        Sx(i8),
        Sy(i8),
        Sz(i8)
    }

    pub fn sx<R: RealScalar>() -> Op<Complex<R>>{
        Op::from_shape_vec((2,2),
            vec![Complex::zero(), Complex::one(),
                    Complex::one(), Complex::zero()]
        ).unwrap()
    }

    pub fn sy<R: RealScalar>() -> Op<Complex<R>>{
        Op::from_shape_vec((2,2),
            vec![Complex::zero(), -Complex::i(),
                    Complex::i(),   Complex::zero()]
        ).unwrap()
    }

    pub fn sz<R: RealScalar>() -> Op<Complex<R>>{
        Op::from_shape_vec((2,2),
            vec![Complex::one(), Complex::zero(),
                    Complex::zero(), -Complex::one()]
        ).unwrap()
    }

    fn eig_generic<R: RealScalar>(val: i8, plu: &[Complex<R>], min: &[Complex<R>]) -> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R> + ComplexField<RealField=R>
    {
        if val > 0{
            let mut vec = Ket::from(plu.to_owned());
            let n = vec.mapv(|x| Complex::norm_sqr(&x)).sum();
            let n = ComplexField::sqrt(n);
            vec /= Complex::from(n);
            //vec.normalize_mut();
            vec
        } else if val < 0 {
            let mut vec = Ket::from(min.to_owned());
            let n = vec.mapv(|x| Complex::norm_sqr(&x)).sum();
            let n = ComplexField::sqrt(n);
            vec /= Complex::from(n);
            //vec.normalize_mut();
            vec
        } else {
            Ket::zeros(2)
        }
    }

    pub fn sx_eig<R: RealScalar>(val: i8)-> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        eig_generic(val, &[Complex::one(), Complex::one()],
                    &[Complex::one(), -Complex::one()])
    }


    pub fn sy_eig<R: RealScalar>(val: i8)-> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        eig_generic(val, &[Complex::i(), Complex::i()],
                    &[Complex::i(), -Complex::i()])
    }

    pub fn sz_eig<R: RealScalar>(val: i8)-> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        eig_generic(val, &[Complex::one(), Complex::zero()],
                    &[Complex::zero(), -Complex::one()])
    }

    pub fn id<R: RealScalar>() -> Op<Complex<R>>
    {
        Op::eye(2)
    }

    fn ket_of<R: RealScalar>(eig: &Eigs) -> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        match eig{
            Eigs::Sx(v) => sx_eig(*v),
            Eigs::Sy(v) => sy_eig(*v),
            Eigs::Sz(v) => sz_eig(*v)
        }
    }

    pub fn auto_ket<R: RealScalar>(eigs: &[Eigs]) -> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        if eigs.len() == 0{
            return Ket::zeros(2);
        }

        let (first, rest) = eigs.split_at(1);
        let mut v = ket_of(&first[0]);
        for eig in rest.iter(){
            v = TensorProd::tensor_ref(&v, &ket_of(eig));
        }

        v
    }

    pub fn ket_from_bitstring<R: RealScalar>(bits: &[i8]) -> Ket<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        if bits.len() == 0{
            return Ket::zeros(2);
        }
        let eigs = Vec::from_iter(bits.iter()
            .map(
            |b| Eigs::Sz(*b)
        ));

        auto_ket(&eigs)
    }

    pub fn n_local_pauli<R: RealScalar>(ops: &[(u8, Op<Complex<R>>)], len: u8) -> Op<Complex<R>>
    where Complex<R> : ComplexScalar<R=R>
    {
        let mut tensor_ops :Vec<Op<Complex<R>>> = Vec::new();
        for k in 0..len{
            match ops.iter().find(|pair|pair.0 == k){
                None => tensor_ops.push(id()),
                Some(pair) => tensor_ops.push(pair.1.clone())
            }
        }
        tensor_list(&tensor_ops)
    }

}

pub mod matrix{
    use std::iter::FromIterator;

    use crate::{RealField, ComplexField};
    use num_traits::{Zero, One};
    use num_complex::Complex;
    use crate::base::quantum::*;
    use qrs_core::reps::matrix::*;
    use crate::ComplexScalar;
    //use nalgebra::DMatrix;


    pub enum Eigs{
        Sx(i8),
        Sy(i8),
        Sz(i8)
    }

    pub fn sx<N: RealField>() -> Op<Complex<N>>{
        Op::from_row_slice(2,2,
                           &[Complex::zero(), Complex::one(),
                               Complex::one(), Complex::zero()]
        )
    }

    pub fn sy<N: RealField>() -> Op<Complex<N>>{
        Op::from_row_slice(2,2,
                           &[Complex::zero(), - Complex::i(),
                               Complex::i(),   Complex::zero()]
        )
    }

    pub fn sz<N: RealField>() -> Op<Complex<N>>{
        Op::from_row_slice(2,2,
                           &[Complex::one(), Complex::zero(),
                               Complex::zero(), -Complex::one()]
        )
    }

    fn eig_generic<N: RealField>(val: i8, plu: &[Complex<N>], min: &[Complex<N>]) -> Ket<Complex<N>>
    where Complex<N> : ComplexField<RealField=N>
    {
        if val > 0{
            let mut vec = Ket::from_row_slice(plu);
            vec.normalize_mut();
            vec
        } else if val < 0 {
            let mut vec = Ket::from_row_slice(min);
            vec.normalize_mut();
            vec
        } else {
            Ket::zeros(2)
        }
    }

    pub fn sx_eig<N: RealField>(val: i8)-> Ket<Complex<N>>{
        eig_generic(val, &[Complex::one(), Complex::one()],
                    &[Complex::one(), -Complex::one()])
    }


    pub fn sy_eig<N: RealField>(val: i8)-> Ket<Complex<N>>{
        eig_generic(val, &[Complex::i(), Complex::i()],
                    &[Complex::i(), -Complex::i()])
    }

    pub fn sz_eig<N: RealField>(val: i8)-> Ket<Complex<N>>{
        eig_generic(val, &[Complex::one(), Complex::zero()],
                    &[Complex::zero(), -Complex::one()])
    }

    pub fn id<N: RealField>() -> Op<Complex<N>>{
        Op::identity(2, 2)
    }

    fn ket_of<N: RealField>(eig: &Eigs) -> Ket<Complex<N>>{
        match eig{
            Eigs::Sx(v) => sx_eig(*v),
            Eigs::Sy(v) => sy_eig(*v),
            Eigs::Sz(v) => sz_eig(*v)
        }
    }

    pub fn auto_ket<N: RealField>(eigs: &[Eigs]) -> Ket<Complex<N>>
        where Complex<N> : ComplexScalar
    {
        if eigs.len() == 0{
            return Ket::zeros(2);
        }

        let (first, rest) = eigs.split_at(1);
        let mut v = ket_of(&first[0]);
        for eig in rest.iter(){
            v = TensorProd::tensor_ref(&v, &ket_of(eig));
        }

        v
    }

    pub fn ket_from_bitstring<N:RealField>(bits: &[i8]) -> Ket<Complex<N>>
        where Complex<N> : ComplexScalar
    {
        if bits.len() == 0{
            return Ket::zeros(2);
        }
        let eigs = Vec::from_iter(bits.iter()
            .map(
                |b| Eigs::Sz(*b)
            ));

        auto_ket(&eigs)
    }

    pub fn n_local_pauli<N:RealField>(ops: &[(u8, Op<Complex<N>>)], len: u8) -> Op<Complex<N>>
        where Complex<N> : ComplexScalar
    {
        let mut tensor_ops :Vec<Op<Complex<N>>> = Vec::new();
        for k in 0..len{
            match ops.iter().find(|pair|pair.0 == k){
                None => tensor_ops.push(id()),
                Some(pair) => tensor_ops.push(pair.1.clone())
            }
        }
        tensor_list(&tensor_ops)
    }

}
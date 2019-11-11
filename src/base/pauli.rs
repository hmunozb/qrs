
pub mod dense{
    use std::iter::FromIterator;

    use alga::general::{RealField};
    use num_traits::{Zero, One};
    use num_complex::Complex;
    use crate::base::quantum::*;
    use crate::base::dense::*;
    //use nalgebra::DMatrix;


    pub enum Eigs{
        Sx(i8),
        Sy(i8),
        Sz(i8)
    }

    pub fn sx<N: RealField>() -> Op<N>{
        Op::from_row_slice(2,2,
            &[Complex::zero(), Complex::one(),
            Complex::one(), Complex::zero()]
        )
    }

    pub fn sy<N: RealField>() -> Op<N>{
        Op::from_row_slice(2,2,
            &[Complex::zero(), - Complex::i(),
            Complex::i(),   Complex::zero()]
        )
    }

    pub fn sz<N: RealField>() -> Op<N>{
        Op::from_row_slice(2,2,
            &[Complex::one(), Complex::zero(),
            Complex::zero(), -Complex::one()]
        )
    }

    fn eig_generic<N: RealField>(val: i8, plu: &[Complex<N>], min: &[Complex<N>]) -> Ket<N>{
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

    pub fn sx_eig<N: RealField>(val: i8)-> Ket<N>{
        eig_generic(val, &[Complex::one(), Complex::one()],
                    &[Complex::one(), -Complex::one()])
    }


    pub fn sy_eig<N: RealField>(val: i8)-> Ket<N>{
        eig_generic(val, &[Complex::i(), Complex::i()],
                    &[Complex::i(), -Complex::i()])
    }

    pub fn sz_eig<N: RealField>(val: i8)-> Ket<N>{
        eig_generic(val, &[Complex::one(), Complex::zero()],
                    &[Complex::zero(), -Complex::one()])
    }

    pub fn id<N: RealField>() -> Op<N>{
        Op::identity(2, 2)
    }

    fn ket_of<N: RealField>(eig: &Eigs) -> Ket<N>{
        match eig{
            Eigs::Sx(v) => sx_eig(*v),
            Eigs::Sy(v) => sy_eig(*v),
            Eigs::Sz(v) => sz_eig(*v)
        }
    }

    pub fn auto_ket<N: RealField>(eigs: &[Eigs]) -> Ket<N>{
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

    pub fn ket_from_bitstring<N:RealField>(bits: &[i8]) -> Ket<N>{
        if bits.len() == 0{
            return Ket::zeros(2);
        }
        let eigs = Vec::from_iter(bits.iter()
            .map(
            |b| Eigs::Sz(*b)
        ));

        auto_ket(&eigs)
    }

    pub fn n_local_pauli<N:RealField>(ops: &[(u8, Op<N>)], len: u8) -> Op<N>{
        let mut tensor_ops :Vec<Op<N>> = Vec::new();
        for k in 0..len{
            match ops.iter().find(|pair|pair.0 == k){
                None => tensor_ops.push(id()),
                Some(pair) => tensor_ops.push(pair.1.clone())
            }
        }
        tensor_list(&tensor_ops)
    }

}
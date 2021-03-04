use qrs_core::quantum::LinearOperator;
use crate::ComplexScalar;
use qrs_core::reps::matrix::{Ket};

/// Represents the sparse linear operator
 ///  sqrt(g) | i > < j |
#[derive(Copy, Clone)]
struct SparseLinOp<N>{
    pub i: u32,
    pub j: u32,
    pub g: N
}

impl<N: ComplexScalar> LinearOperator<Ket<N>, N> for SparseLinOp<N>

{
    fn map(&self, v: &Ket<N>) -> Ket<N> {
        let mut lv : Ket<N> = Ket::zeros(v.shape().0);
        unsafe { *lv.get_unchecked_mut(self.i as usize) =
            *v.get_unchecked(self.j as usize) * self.g; }
        return lv;
    }

    fn conj_map(&self, v: &Ket<N>) -> Ket<N> {
        let mut lv = Ket::zeros(v.shape().0);
        unsafe { *lv.get_unchecked_mut(self.j as usize) =
            *v.get_unchecked(self.i as usize) * self.g.conjugate(); }
        return lv;
    }

    fn positive_map(&self, v: &Ket<N>) -> Ket<N>{
        let mut lv = Ket::zeros(v.shape().0);
        unsafe{
            *lv.get_unchecked_mut(self.j as usize) =
                *v.get_unchecked(self.j as usize) * self.g.modulus_squared().into();
        }
        return lv;
    }

    fn add_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        unsafe{
            *target.get_unchecked_mut(self.i as usize) +=
                *v.get_unchecked(self.j as usize) * self.g;
        }
    }

    fn add_conj_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        unsafe{
            *target.get_unchecked_mut(self.j as usize) +=
                *v.get_unchecked(self.i as usize) * self.g.conjugate();
        }
    }

    fn add_positive_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        unsafe{
            *target.get_unchecked_mut(self.j as usize) +=
                *v.get_unchecked(self.j as usize) * (self.g.modulus_squared().into());
        }
    }

    fn positive_ev(&self, v: &Ket<N>) -> N::R {
        use qrs_core::ComplexField;
        unsafe {
            return (self.g.modulus_squared()) * v.get_unchecked(self.j as usize).modulus_squared();
        }
    }
}

use nalgebra::{DVector, ComplexField};
/// Represents the sparse linear operator
 ///  \sum_i g_ii | i > < i |
#[derive(Clone)]
struct DiagonalLinOp<N: ComplexScalar>{
    diag: Ket<N>
}

impl<N: ComplexScalar> LinearOperator<Ket<N>, N> for DiagonalLinOp<N>{
    fn map(&self, v: &Ket<N>) -> Ket<N> {
        DVector::from_iterator(v.shape().0, v.iter().zip(self.diag.iter())
                .map(|(&vi,&a)| vi * a))
    }

    fn conj_map(&self, v: &Ket<N>) -> Ket<N> {
        DVector::from_iterator(v.shape().0, v.iter().zip(self.diag.iter())
            .map(|(&vi,&a)| vi * a.conjugate()))
    }

    fn positive_map(&self, v: &Ket<N>) -> Ket<N> {
        DVector::from_iterator(v.shape().0, v.iter().zip(self.diag.iter())
            .map(|(&vi,&a)| vi * a.modulus_squared().into()))
    }

    fn add_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        target.iter_mut().zip(v.iter().zip(self.diag.iter()))
            .map(|(t, (&vi, &a))| *t += vi*a);
    }

    fn add_conj_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        target.iter_mut().zip(v.iter().zip(self.diag.iter()))
            .map(|(t, (&vi, &a))| *t += vi*a.conjugate());
    }

    fn add_positive_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        target.iter_mut().zip(v.iter().zip(self.diag.iter()))
            .map(|(t, (&vi, &a))| *t += vi*a.modulus_squared().into());
    }

    fn positive_ev(&self, v: &Ket<N>) -> <N as ComplexScalar>::R {
        use num_traits::Zero;
        v.iter().zip(self.diag.iter())
            .fold(N::R::zero(),
                  |acc, (&vi, &a)| acc + vi.modulus_squared() * a.modulus_squared())
    }
}
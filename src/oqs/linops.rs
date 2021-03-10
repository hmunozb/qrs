use qrs_core::quantum::LinearOperator;
use crate::ComplexScalar;
use qrs_core::reps::matrix::{Ket};

/// Represents the sparse linear operator
 ///  sqrt(g) | i > < j |
#[derive(Copy, Clone)]
pub struct SparseLinOp<N>{
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
pub struct DiagonalLinOp<N: ComplexScalar>{
    pub diag: Ket<N>
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
        for (t, (&vi, &a)) in target.iter_mut().zip(v.iter().zip(self.diag.iter())){
            *t += vi * a;
        }
    }

    fn add_conj_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        for (t, (&vi, &a)) in target.iter_mut().zip(v.iter().zip(self.diag.iter())){
            *t += vi*a.conjugate()
        }
    }

    fn add_positive_map_to(&self, v: &Ket<N>, target: &mut Ket<N>) {
        for (t, (&vi, &a)) in  target.iter_mut().zip(v.iter().zip(self.diag.iter())){
            *t += vi*a.modulus_squared().into()
        }
    }

    fn positive_ev(&self, v: &Ket<N>) -> <N as ComplexScalar>::R {
        use num_traits::Zero;
        v.iter().zip(self.diag.iter())
            .fold(N::R::zero(),
                  |acc, (&vi, &a)| acc + vi.modulus_squared() * a.modulus_squared())
    }
}

#[derive(Copy, Clone)]
pub enum CombinedLinOp<L1, L2>{
    L1(L1),
    L2(L2),
}

impl<V, N: ComplexScalar, L1, L2> LinearOperator<V, N> for CombinedLinOp<L1, L2>
where  L1: LinearOperator<V,N>, L2:LinearOperator<V, N>
{
    fn map(&self, v: &V) -> V {
        match self{
            Self::L1(l)=> l.map(v),
            Self::L2(l) => l.map(v)
        }
    }

    fn conj_map(&self, v: &V) -> V {
        match self{
            Self::L1(l)=> l.conj_map(v),
            Self::L2(l) => l.conj_map(v)
        }
    }

    fn positive_map(&self, v: &V) -> V {
        match self{
            Self::L1(l)=> l.positive_map(v),
            Self::L2(l) => l.positive_map(v)
        }
    }

    fn add_map_to(&self, v: &V, target: &mut V) {
        match self{
            Self::L1(l)=> l.add_map_to(v, target),
            Self::L2(l) => l.add_map_to(v, target)
        }
    }

    fn add_conj_map_to(&self, v: &V, target: &mut V) {
        match self{
            Self::L1(l)=> l.add_conj_map_to(v, target),
            Self::L2(l) => l.add_conj_map_to(v, target)
        }
    }

    fn add_positive_map_to(&self, v: &V, target: &mut V) {
        match self{
            Self::L1(l)=> l.add_positive_map_to(v, target),
            Self::L2(l) => l.add_positive_map_to(v, target)
        }
    }

    fn positive_ev(&self, v: &V) -> <N as ComplexScalar>::R {
        match self{
            Self::L1(l)=> l.positive_ev(v),
            Self::L2(l) => l.positive_ev(v)
        }
    }
}
use std::collections::btree_map::{BTreeMap, Entry};
use std::iter::FromIterator;
use std::marker::PhantomData;

use crate::algebra::free_algebra::{DynAlgebraOne, DynFreeAlgebra, DynVectorSpace, DynVectorSpaceZero};
use crate::ComplexScalar;
use crate::quantum::{ConjugatingWrapper, QBra, QKet, QObj, QOp, QOpType, QRep, QType};

//use num_traits::{Zero, One};

/// Represents a single product of bosonic operators in normal form
/// i.e. each mapping  (i, (nr, nl)) represents the operation of lowering the number state i by nl
/// followed by raising it by nr
/// Note that no entry for i is implicitly (i, (0, 0)), which is an identity operation
/// A zero operation is represented by None
#[derive(Clone, PartialEq)]
pub struct NormalBosonic{
    b: Option<BTreeMap<u16, (u16, u16)>>
}
//
//impl PartialEq for NormalBosonic{
//    fn eq(&self, other: &Self) -> bool {
//        if let Some(b1) = &self.b{
//            if let Some(b2) = &other.b {
//                b1 == b2
//            } else {
//                false
//            }
//        } else { // self is None
//            if let &None = other.b{
//                true
//            } else {
//                false
//            }
//        }
//    }
//}

impl DynVectorSpaceZero for NormalBosonic{
    fn zero() -> Self{
        NormalBosonic{b: None}
    }
    fn is_zero(&self) -> bool {
        if let &None = &self.b {
            true
        } else {
            false
        }
    }
}

impl DynAlgebraOne for NormalBosonic{
    fn one() -> Self{
        let b = BTreeMap::new();
        NormalBosonic{b: Some(b)}
    }

    fn is_one(&self) -> bool
    where Self: PartialEq, {
        if let Some(b) = &self.b{
            if b.is_empty(){
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl NormalBosonic{
    pub fn from_iter<T: IntoIterator<Item = (u16, (u16, u16))>>(iter: T)-> Self{
        let b = BTreeMap::from_iter(iter);
        NormalBosonic{b: Some(b)}
    }
    pub fn apply_to_basis_ket(&self, u: BosonicBasisKet) -> BosonicBasisKet{
        let mut u = u;

        match &self.b{
            None => {
                u.b = None;
            },
            Some(b) =>{
                for (&i,&(r, l)) in b.iter(){
                    u.lower_n(i, l);
                    u.raise_n(i, r);
                }
            }
        }

        u
    }

    pub fn apply_to_basis_ket_ref(&self, u: &BosonicBasisKet) -> BosonicBasisKet{
        let mut u = u.clone();
        self.apply_to_basis_ket(u)
    }

    pub fn apply_to_ket<T: ComplexScalar>(&self, mut u: BosonicKet<T>) -> BosonicKet<T>{
        u.linear_map(&|ui| self.apply_to_basis_ket(ui), &|t| t)
    }

    pub fn apply_to_ket_ref<T: ComplexScalar>(&self, u: &BosonicKet<T>) -> BosonicKet<T>{
        u.linear_map_ref(&|ui| self.apply_to_basis_ket_ref(ui), &|&t| t)
    }

//    pub fn normal_form_product(a1: Self, a2: Self) {
//        if a1.b.is_empty() || a2.b.is_empty(){
//            Self::zero()
//        }
//
//        let mut i1 = a1.b.iter().peekable();
//        let mut i2 = a1.b.iter().peekable();
//        loop{
//
//            let k1 = match i1.peek(){None => Break, Some(&k) => k};
//            let k2 = match i2.peek(){None => Break, Some(&k) => k};
//            if k1.0 < k2.0{
//                i1.next();
//            } else if k2.0 < k1.0 {
//                i2.next();
//            } else { //k1 == k2
//                let &n1 = k1.1;
//                let &n2 = k2.1;
//
//            }
//        }
//
//
//    }
}

/// A Bosonic basis state simply counts the particle number of each index
/// A vacuum bosonic is canonically represented by an empty BTreeMap
/// An annihilated bosonic state is represented by None
///
/// Number lowering will always remove a zero-particle index from the map or
/// annihilate the state as appropriate. Applying raise_n and lower_n with n=0 is a no-op.
/// Thus, the only entries in the map are indices with non-zero particle numbers.
/// This guarantees that two basis kets are equal if and only if
/// their map entries are the same, precisely the PartialEq condition of BTreeMap
#[derive(Clone, PartialEq)]
pub struct BosonicBasisKet{
    b: Option<BTreeMap<u16, u16>>
}
//pub struct BosonicBasisBra{
//    b: BTreeMap<u16, u16>
//}

impl BosonicBasisKet{
    pub fn vacuum() -> Self{
        BosonicBasisKet{b: Some(BTreeMap::new())}
    }

    pub fn raise(&mut self, i: u16){
        if let Some(b) = &mut self.b{
            *b.entry(i).or_insert(0) += 1;
        }
    }

    pub fn raise_n(&mut self, i: u16, n: u16){
        if n == 0 {return};

        if let Some(b) = &mut self.b{
            *b.entry(i).or_insert(0) += n;
        }
    }

    pub fn lower(&mut self, i: u16){
        let mut annihilate = false;
        if let Some(b) = &mut self.b{
            match b.entry(i){
                Entry::Occupied(mut e) => {
                    let &k = e.get();
                    if k > 1{
                        *e.get_mut() -= 1;
                    } else if k == 1{
                        e.remove_entry();
                    } else{
                        annihilate = true;
                    }
                }
                Entry::Vacant(_e) => {
                    annihilate = true;
                }
            }
        }

        if annihilate{
            self.b = None;
        }
    }

    pub fn lower_n(&mut self, i: u16, n: u16){
        if n == 0 {return};

        let mut annihilate = false;
        if let Some(b) = &mut self.b{
            match b.entry(i){
                Entry::Occupied(mut e) => {
                    let &k = e.get();
                    if k > n{
                        *e.get_mut() -= n;
                    } else if k == n{
                        e.remove_entry();
                    } else{
                        annihilate = true;
                    }
                }
                Entry::Vacant(_e) => {
                    annihilate = true;
                }
            }
        }

        if annihilate{
            self.b = None;
        }
    }

    /// Dots the two basis kets in a canonical manner
    /// If either ket is none (annihilated) the dot is zero
    /// otherwise, compare whether the quantum numbers are equal
    fn dot(&self, other: &Self) -> bool{
        self.b.as_ref().map_or(
            false,
            |t| other.b.as_ref().map_or(
                   false,
                    |t2| t == t2
               )
        )
    }

}


type BosonicKet<T> = DynVectorSpace<BosonicBasisKet, T>;
type BosonicBra<T> = ConjugatingWrapper<BosonicKet<T>>;
type BasisKet = BTreeMap<u16, u16>;
type BosonicOp<T> = DynFreeAlgebra<NormalBosonic, T>;

#[derive(Clone)]
pub struct BosonicQRep<T>{
    _phantom: PhantomData<T>
}

impl<T: ComplexScalar> QRep<T> for BosonicQRep<T>{
    type KetRep = BosonicKet<T>;
    type BraRep = BosonicBra<T>;
    type OpRep = BosonicOp<T>;

    fn qbdot(bra: &Self::BraRep, ket: &Self::KetRep) -> T {
        bra.q.evaluate_ref(
            &|ui|{ ket.evaluate_ref(
                &|vi| if ui.dot(vi) {T::one()} else {T::zero()},
                &|&t| t,
                &T::zero() )},
            &|&t| t,
            &T::zero()
        )
    }

    fn qdot(u: &Self::KetRep, v: &Self::KetRep) -> T {
        u.evaluate_ref(
            &|ui|{ v.evaluate_ref(
                &|vi| if ui.dot(vi) {T::one()} else {T::zero()},
                 &|&t| t,
                &T::zero() )},
            &|t| t.conjugate(),
            &T::zero()
        )
    }


    fn khemv(op: &BosonicOp<T>, alpha: T, x: &Self::KetRep, y: &mut Self::KetRep, beta: T) {
        if beta.is_zero(){

            // *y = op.evaluate_ref(&|xi| DynVectorSpace::Element(xi.apply_to_ket_ref(x)),
            //                      &|&t| t,
            //                      &DynVectorSpace::Zero,
            //                     &DynVectorSpace::Element(x.clone()))
        }
    }
}

impl<T: ComplexScalar> QObj<T> for BosonicKet<T>{
    type Rep = BosonicQRep<T>;
    type Dims = ();

    fn qdim(&self) -> Self::Dims {
        ()
    }

    fn qtype(&self) -> QType {
        QType::QKet
    }

    fn qaxpy(&mut self, a: T, x: &Self) {
        let ax = x.clone() * a;
        *self += ax;
    }

    fn qscal(&mut self, a: T) {
        *self *= a;
    }

    fn qaxby(&mut self, a: T, x: &Self, b: T) {
        *self *= b;
        let ax = x.clone() * a;
        *self += ax;
    }
}

impl<T: ComplexScalar> QKet<T> for BosonicKet<T> { }
impl<T: ComplexScalar> QBra<T> for BosonicBra<T> { }

impl<T: ComplexScalar> QObj<T> for BosonicOp<T>{
    type Rep = BosonicQRep<T>;
    type Dims = ();

    fn qdim(&self) -> Self::Dims {
        ()
    }

    fn qtype(&self) -> QType {
        QType::QOp(QOpType::Ge)
    }

    fn qaxpy(&mut self, a: T, x: &Self) {
        let ax = x.clone() * a;
        *self += ax;
    }

    fn qscal(&mut self, a: T) {
        *self *= a;
    }

    fn qaxby(&mut self, a: T, x: &Self, b: T) {
        *self *= b;
        let ax = x.clone() * a;
        *self += ax;
    }
}

impl<T: ComplexScalar> QOp<T> for BosonicOp<T>{ }

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_bosonic(){
        let vac = NormalBosonic::zero();
        let a1 = NormalBosonic::from_iter(vec![(0,(2,3)), (1, (1,4))]);
    }

}
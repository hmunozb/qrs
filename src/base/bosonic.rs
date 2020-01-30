

use std::collections::btree_map::{BTreeMap, Entry};
use crate::algebra::free_algebra::{DynFreeAlgebra, DynVectorSpace, DynVectorSpaceZero, DynAlgebraOne};
use std::iter::FromIterator;

//use num_traits::{Zero, One};

/// Represents a single product of bosonic operators in normal form
/// i.e. each mapping  (i, (nr, nl)) represents the operation of lowering the number state i by nl
/// followed by raising it by nr
/// Note that no entry for i is implicitly (i, (0, 0)), which is an identity operation
/// A zero operation is represented by None
#[derive(PartialEq)]
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
    pub fn apply_to_basis(&self, u: BosonicBasisKet) -> BosonicBasisKet{
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
/// their map entries are the same
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
}

type BosonicKet<T> = DynVectorSpace<BosonicBasisKet, T>;

type BasisKet = BTreeMap<u16, u16>;
type Op<T> = DynFreeAlgebra<NormalBosonic, T>;
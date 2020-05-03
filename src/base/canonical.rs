use crate::algebra::free_algebra::{//DynFreeAlgebra,
                                   DynVectorSpace};

pub struct CanonicalBasisPKet{
    p: i64
}
type PKet<T> = DynVectorSpace<CanonicalBasisPKet, T>;

pub struct PShift{
    dp: i64
}
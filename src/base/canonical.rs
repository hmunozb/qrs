use crate::algebra::free_algebra::{//DynFreeAlgebra,
                                   DynVectorSpace};

pub struct CanonicalBasisPKet{
    p: i64
}
type PKet<T> = DynVectorSpace<CanonicalBasisPKet, T>;

pub struct PShift{
    dp: i64
}

//impl PShift{
//    fn dot<T: Sized>(&self, ket: PKet<T>) -> PKet<T>{
//        let f = |k: CanonicalBasisPKet| CanonicalBasisPKet{p: k.p + self.dp};
//        ket.linear_apply(&f)
//    }
//}
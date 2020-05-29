use std::borrow::{Cow, Borrow};
use nalgebra::dimension::DimName;
use generic_array::{GenericArray, ArrayLength};
use std::ops::{Add, AddAssign, SubAssign, MulAssign};
use std::marker::PhantomData;
use itertools::Itertools;
use num_traits::Unsigned;
use ndarray::Dim;

pub mod storage;
pub mod stride;
pub use storage::*;

// pub trait StridedStorage<N>{
//     fn strided_iter(&self)
// }



pub struct LinearSpaceVector<N, S: Storage<N>>{
    data: S,
    _phantom: PhantomData<N>
}

impl<N, S: Storage<N>> LinearSpaceVector<N, S>{
    pub fn from_data(data: S) -> LinearSpaceVector<N, S>{
        LinearSpaceVector{data, _phantom: PhantomData}
    }
    pub fn to_owned(&self) -> LinearSpaceVector<N, S::OwnedStorage>{
        let owned = self.data.to_owned();
        LinearSpaceVector{data: owned, _phantom: PhantomData}
    }
}

impl<N: Copy, S: StorageMut<N>, S2> AddAssign<&LinearSpaceVector<N, S2>> for LinearSpaceVector<N, S>
where S2: Storage<N>, N: AddAssign{
    fn add_assign(&mut self, rhs: &LinearSpaceVector<N, S2>) {
        for (a, &b) in self.data.mem_iter_mut().zip_eq(rhs.data.mem_iter()){
            *a += b;
        }
    }
}

impl<N: Copy, S: StorageMut<N>, S2> SubAssign<&LinearSpaceVector<N, S2>> for LinearSpaceVector<N, S>
    where S2: Storage<N>, N: SubAssign{
    fn sub_assign(&mut self, rhs: &LinearSpaceVector<N, S2>) {
        for (a, &b) in self.data.mem_iter_mut().zip_eq(rhs.data.mem_iter()){
            *a -= b;
        }
    }
}

impl<N: Copy, S: StorageMut<N>> MulAssign<N> for LinearSpaceVector<N, S>
    where N: MulAssign{
    fn mul_assign(&mut self, rhs: N) {
        for a in self.data.mem_iter_mut(){
            *a *= rhs;
        }
    }
}

impl<N: Copy, S: Storage<N>, S2> Add<&LinearSpaceVector<N, S2>> for &LinearSpaceVector<N, S>
where S2: Storage<N>, N: AddAssign
{
    type Output = LinearSpaceVector<N, S::OwnedStorage>;

    fn add(self, rhs: &LinearSpaceVector<N, S2>) -> LinearSpaceVector<N, S::OwnedStorage> {
        let mut sum = self.to_owned();
        sum += rhs;
        sum
    }
}

// impl< N> Borrow<SliceStorage<'_, N>> for VecStorage<N>
// {
//     fn borrow<'a>(&'a self) -> &'a SliceStorage<'a, N> {
//         &SliceStorage(& self.0.borrow())
//     }
// }

// impl<'a, N: Copy> ToOwned for &SliceStorage<'a, N>{
//     type Owned = VecStorage<N>;
//
//     fn to_owned(&self) -> VecStorage<N> {
//         VecStorage(self.0.to_owned())
//     }
// }

// impl<'a, N: Copy> ToOwned for SliceMutStorage<'a, N>{
//     type Owned = VecStorage<N>;
//
//     fn to_owned(&self) -> VecStorage<N> {
//         VecStorage(self.0.to_owned())
//     }
// }

#[cfg(test)]
pub mod tests{

}
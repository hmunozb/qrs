use generic_array::*;
use std::borrow::{Cow, Borrow};

/// Generic array storage (Borrow type = Owned type = Array [ N; D])
#[derive(Clone)]
pub struct ArrayStorage<N, D: ArrayLength<N>>(GenericArray<N, D>);
/// Copy-on-write Vec storage (Borrow type = slice, owned slice = Vec)
#[derive(Clone)]
pub struct CowVecStorage<'a, N: Copy>(Cow<'a, [N]>);
/// Owned vector storage
#[derive(Clone)]
pub struct VecStorage<N>(Vec<N>);
/// Slice reference storage
pub struct SliceStorage<'a, N>(&'a [N]);
/// Mutable slice reference storage
pub struct SliceMutStorage<'a, N>(&'a mut [N]);

/// Trait implemented by all contiguous memory storage structures
pub trait Storage<N>{
    type OwnedStorage: OwnedStorage<N>;

    /// Borrow the storage as a SliceStorage
    fn borrow_slice(&self) -> SliceStorage<N>;
    /// Create an OwnedStorage out of the Storage ref
    fn to_owned(&self) -> Self::OwnedStorage;
    /// Convert this storage into its associated owned storage
    /// In particular, ensures that CovVec Storage becomes an owned
    /// VecStorage
    fn into_owned(self) -> Self::OwnedStorage;
    fn mem_iter(&self) -> std::slice::Iter<N>;
}
/// Contiguous memory structure with mutable access
pub trait StorageMut<N>: Storage<N>{
    fn borrow_slice_mut(&mut self) -> SliceMutStorage<N>;
    fn mem_iter_mut(&mut self) -> std::slice::IterMut<N>;
}

/// A storage structure is `OwnedStorage` if and only if it is a clonable
/// resource of mutable contiguous storage
pub trait OwnedStorage<N>: StorageMut<N> + Clone{ }
impl<T, N> OwnedStorage<N> for T where T: StorageMut<N> + Clone{ }


/*
 *
 * Implementations
 *
 */

impl<N: Copy, D: ArrayLength<N>> Storage<N> for ArrayStorage<N, D>{
    type OwnedStorage = Self;
    fn borrow_slice(&self) -> SliceStorage< N> {
        SliceStorage(self.0.as_slice())
    }
    fn to_owned(&self) -> Self {
        self.clone()
    }
    fn into_owned(self) -> Self {
        self
    }
    fn mem_iter(&self) -> std::slice::Iter<N>{
        self.0.iter()
    }
}
impl<N: Copy, D: ArrayLength<N>> StorageMut<N> for ArrayStorage<N, D>{
    fn borrow_slice_mut(&mut self) -> SliceMutStorage<N> {
        SliceMutStorage(self.0.as_mut_slice())
    }

    fn mem_iter_mut(&mut self) -> std::slice::IterMut<N> {
        self.0.iter_mut()
    }
}
impl<'a, N: Copy> Storage<N> for CowVecStorage<'a, N>{
    type OwnedStorage = VecStorage<N>;

    fn borrow_slice(&self) -> SliceStorage<N> {
        SliceStorage(self.0.borrow())
    }
    fn to_owned(&self) -> VecStorage<N> {
        VecStorage(self.0.to_vec())
    }
    fn into_owned(self) -> VecStorage<N> {
        VecStorage(self.0.into_owned())
    }
    fn mem_iter(&self) -> std::slice::Iter<N>{
        self.0.iter()
    }
}

impl<'a, N: Copy> StorageMut<N> for CowVecStorage<'a, N>{
    fn borrow_slice_mut(&mut self) -> SliceMutStorage<N> {
        let slm = self.0.to_mut();
        SliceMutStorage(slm.as_mut_slice())
    }
    fn mem_iter_mut(&mut self) -> std::slice::IterMut<N>{
        self.0.to_mut().iter_mut()
    }
}

impl<N: Copy> Storage<N> for VecStorage<N>{
    type OwnedStorage = Self;

    fn borrow_slice(&self) -> SliceStorage<N>{
        SliceStorage(&self.0)
    }
    fn to_owned(&self) -> VecStorage<N>{
        self.clone()
    }
    fn into_owned(self) -> VecStorage<N>{
        self
    }
    fn mem_iter(&self) -> std::slice::Iter<N>{
        self.0.iter()
    }
}
impl<N: Copy> StorageMut<N> for VecStorage<N>{
    fn borrow_slice_mut(&mut self) -> SliceMutStorage<N>{
        SliceMutStorage(&mut self.0)
    }
    fn mem_iter_mut(&mut self) -> std::slice::IterMut<N>{
        self.0.iter_mut()
    }
}

impl<'a, N: Copy> Storage<N> for SliceStorage<'a, N>{
    type OwnedStorage = VecStorage<N>;

    fn borrow_slice(&self) -> SliceStorage<N>{
        SliceStorage(self.0)
    }
    fn to_owned(&self) -> VecStorage<N>{
        VecStorage(self.0.to_owned())
    }
    fn into_owned(self) -> VecStorage<N>{
        VecStorage(self.0.to_owned())
    }
    fn mem_iter(&self) -> std::slice::Iter<N>{
        self.0.iter()
    }
}

impl<'a, N: Copy> Storage<N> for SliceMutStorage<'a, N>{
    type OwnedStorage = VecStorage<N>;

    fn borrow_slice(&self) -> SliceStorage<N>{
        SliceStorage(self.0)
    }
    fn to_owned(&self) -> VecStorage<N>{
        VecStorage(self.0.to_owned())
    }
    fn into_owned(self) -> VecStorage<N>{
        VecStorage(self.0.to_owned())
    }
    fn mem_iter(&self) -> std::slice::Iter<N>{
        self.0.iter()
    }
}

impl<'a, N: Copy> StorageMut<N> for SliceMutStorage<'a, N>{
    fn borrow_slice_mut(&mut self) -> SliceMutStorage<N>{
        SliceMutStorage(self.0)
    }
    fn mem_iter_mut(&mut self) -> std::slice::IterMut<N>{
        self.0.iter_mut()
    }
}
use super::storage::*;
use num_traits::{Signed, NumAssignOps};
use std::marker::PhantomData;
use std::ops::AddAssign;


pub struct StridedStorage<'a, N, S: Storage<N>, Sd: StrideData>
{
    data: &'a S,
    stride: Sd,
    _phantom: PhantomData<N>
}
pub struct StridedIter<'a, N :'a, Sd: StrideData>{
    it: std::slice::Iter<'a, N>,
    stride_iter: <Sd as StrideData>::StrideIter,
    _phantom: PhantomData<Sd>
}
impl<'a, N: 'a, Sd: StrideData> StridedIter<'a, N, Sd>{
    pub fn new<S: Storage<N>>(s: &'a S, stride: Sd) -> StridedIter<'a, N, Sd>{
        //let sl = s.borrow_slice();
        let it = s.mem_iter();
        let stride_iter = stride.iter();

        StridedIter{it, stride_iter, _phantom: PhantomData}
    }
}
// impl<'a, N: 'a, Sd: StrideData> Iterator for StridedIter<'a, N, Sd>{
//     type Item = &'a N;
//
//     fn next(&mut self) -> Option<&'a N>{
//         let (_, delta)  = self.stride_iter.next()?;
//         self.it.nth(delta-1 as usize)
//     }
// }

pub struct StridedIterMut<'a, N :'a, Sd: StrideData>{
    it: std::slice::IterMut<'a, N>,
    stride_iter: <Sd as StrideData>::StrideIter,
    _phantom: PhantomData<Sd>
}

// impl<'a, N: 'a, Sd: StrideData> Iterator for StridedIterMut<'a, N, Sd>{
//     type Item = &'a mut N;
//
//     fn next(&mut self) -> Option<&'a mut N>{
//         let (_, delta) : (usize, usize) = self.stride_iter.next()?;
//         self.it.nth(delta-1)
//     }
// }

pub trait StrideData<I: Signed = isize>{
    type StrideIter: Iterator<Item=(I, I)>;
    type Idx;
    fn get(&self, idx: Self::Idx) -> Option<I>;
    fn iter(&self) -> Self::StrideIter;
}

pub struct ContiguousIter<'a, N: 'a>(std::slice::Iter<'a, N>);
pub struct ContiguousIterMut<'a, N: 'a>(std::slice::IterMut<'a, N>);
impl<'a, N: 'a> Iterator for ContiguousIter<'a, N>{
    type Item = &'a N;

    fn next(& mut self) -> Option<&'a N>{
        self.0.next()
    }
}

pub struct Stride1D<I: Signed>{pub n: I, pub dn: I}
impl<I: Signed> Stride1D<I>{
    pub fn new(n: I, dn: I) -> Self{
        //assert!(n >= 0);
        //assert_eq!(n.signum(), dn.signum());
        Stride1D{n, dn}
    }
    pub fn new_contiguous(n: I)-> Self{
        Self::new(n, I::one())
    }
}
impl<I: Signed + Copy + PartialOrd + AddAssign> StrideData<I> for Stride1D<I>{
    type StrideIter = Stride1DIter<I>;
    type Idx = I;
    #[inline(always)]
    fn get(&self, idx: I) -> Option<I>{
        if idx < self.n {
            let p = idx * self.dn;
            Some(p)
        } else {
            None
        }
    }

    fn iter(&self) -> Stride1DIter<I>{
        Stride1DIter::new(self.n, self.dn)
    }
}
pub struct Stride1DIter<I: Signed>{i: I, n: I, dn: I}
impl<I: Signed + PartialOrd> Stride1DIter<I> {
    pub fn new(n: I, dn: I) -> Self{
        assert!(n >= I::zero());
        //assert_eq!(n.signum(), dn.signum());
        let i = I::zero();
        Stride1DIter{i, n, dn}
    }
}

impl<I: Signed+Copy + PartialOrd + AddAssign> Iterator for Stride1DIter<I>{
    type Item = (I, I);

    fn next(&mut self) -> Option<(I, I)> {
        let i = self.i;
        self.i += I::one();
        if i < self.n {
            let p = i * self.dn;
            Some((p, self.dn))
        } else {
            None
        }
    }
}
pub struct Stride2DIter<I: Signed>{
    i: I, j: I, // logical indices
    di: I, dj: I, // Memory Strides
    ni: I, nj: I, // logical dimensions
    ldi: I,  //leading dimension
}

impl<I: Signed + Copy + PartialOrd> Stride2DIter<I>{
    pub fn new(di: I, dj: I, ni: I, nj: I, ldi: I) -> Self{
        assert!(ni.is_positive());
        assert!(nj.is_positive());
        let i = I::zero();
        let j = I::zero();
        Self{i, j, di, dj, ni, nj,ldi}
    }
}
impl<I: Signed + Copy + PartialOrd + AddAssign> Iterator for Stride2DIter<I>{
    type Item = (I, I);

    fn next(&mut self) -> Option<(I, I)>{
        let i = self.i;
        let j = self.j;
        let mut dp = self.di;
        if j >= self.nj {
            return None;
        }
        self.i += I::one();
        if self.i >= self.ni {
            self.i = I::zero();
            self.j += I::one();
            dp += self.ldi - self.ni*self.di;
        }

        let p = i * self.di + j * self.ldi * self.dj;
        Some((p, dp))

    }
}
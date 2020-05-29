use std::ops::Add;
use num_traits::Zero;

pub trait Sum<A = Self>: Sized{
    fn sum<I : Iterator<Item=A>>(iter: I) -> Self;
}

impl<N> Sum<N> for N
where N: Zero + Add
{
    fn sum<I: Iterator<Item=N>>(iter: I) -> N {
        iter.fold(Zero::zero(), Add::add)
    }
}

impl<'a, N> Sum<&'a N> for N
where N: 'a + Zero, N: Add<&'a N, Output=N>
{
    fn sum<I: Iterator<Item=&'a N>>(iter: I) -> N {
        iter.fold(Zero::zero(), Add::add)
    }
}

// why is this perfect example in the std docs but not actually implemented?
pub fn advance_n_and_return_first<I>(iter: &mut I, total_step: usize) -> Option<I::Item>
    where
        I: Iterator,
{
    let next = iter.next();
    if total_step > 1 {
        iter.nth(total_step-2);
    }
    next
}
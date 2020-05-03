pub use simd_phys;

pub use interp_fn::*;
pub use matrix::*;
pub use partitioned_haml::*;
pub use qrs_core::eig::dense::*;
pub use time_dep_op::*;


mod matrix;
mod interp_fn;
mod partitioned_haml;
mod time_dep_op;
pub mod degen;
pub mod diff;


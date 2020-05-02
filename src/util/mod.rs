//mod eig_resolver_dmatrix;
//pub mod eig_resolver;
mod matrix;
mod interp_fn;
mod partitioned_haml;
mod time_dep_op;
pub mod degen;
pub mod diff;
//pub mod simd;
pub use simd_phys;
//mod complex;

pub use qrs_core::eig::dense::*;
//pub use eig_resolver_dmatrix::*;
pub use matrix::*;
pub use interp_fn::*;
pub use time_dep_op::*;
pub use partitioned_haml::*;
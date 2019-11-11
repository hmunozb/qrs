# qrs: Quantum Rust Library
A Rust library for simulating time-dependent quantum dynamics. 

(Under highly diabatic development)

## Usage

Currently supports dense representation of quantum objects. 
Basic operators such as the Pauli matrices can be created as
```rust
use qrs::base::pauli::dense as pauli;
let sx = pauli::sx::<f64>();
```
which creates a complex number operator `sx` over f64 precision reals.

## Dependencies

This library currently requires system linkage to GSL and lapacke/cblas. On macOS, you can install
`gsl` and `openblas` with homebrew. 

## Building with qrs

To build on macOS with openblas routines: 
* When you create a package that depends on `qrs`, add to your Cargo.toml dependencies
  ````toml
  [dependencies]
  openblas-src = {version="0.7.0", default-features=false, features=["system"]}
  ```` 
  to instruct Cargo that you are linking to qrs
* To let cargo know the location of the libraries, create the file `./.cargo/config/` in the directory of 
your Cargo.toml file with the contents
  ```toml
  [target.x86_64-apple-darwin]
  rustflags = [   "-L", "native=/usr/local/opt/gcc/lib/gcc/9/",
                "-L", "native=/usr/local/opt/openblas/lib"]
  ```
  Alternatively, you can build using other supported bindings/sources for lapacke/cblas on rust. See the
[blas-lapack-rs instructions](https://github.com/blas-lapack-rs/blas-lapack-rs.github.io/wiki). 
If you use a source package, you only to have a prior system installation of GSL. However, lapack/cblas will
be compiled with your crate. 
Normal Linux/Unix targets should not need a config file with standard installation directories

## Planned Features

* Sparse operator representation
* Interface with Python objects and functions


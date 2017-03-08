#![allow(dead_code)]
#![allow(unused_variables)]
#[macro_use]

extern crate rand;

pub mod layer;
pub mod matrix;
pub mod util;
pub mod dataset;
pub mod model;

pub use dataset::DataSet;
pub use model::Model;
pub use layer::FNNLayer;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}

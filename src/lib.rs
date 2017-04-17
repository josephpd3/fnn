#![allow(dead_code)]
#![allow(unused_variables)]
#[macro_use]

extern crate rand;
extern crate statrs;

pub mod layer;
pub mod matrix;
pub mod dataset;
pub mod model;
pub mod loss;
pub mod optimizer;
pub mod util;

pub use dataset::DataSet;
pub use model::Model;
pub use layer::FNNLayer;
pub use util::activation;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}

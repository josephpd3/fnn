#![allow(dead_code)]
#![allow(unused_variables)]
#[macro_use]

extern crate rand;
extern crate statrs;
extern crate rayon;

pub mod layer;
pub mod matrix;
pub mod dataset;
pub mod model;
pub mod loss;
pub mod optimizer;
pub mod util;

pub use dataset::DataSet;
pub use model::{
    Model,
    ModelDescription
};
pub use layer::Layer;
pub use optimizer::FNNOptimizer as Optimizer;
pub use loss::FNNLoss as Loss;
pub use util::activation;

#[cfg(test)]
mod tests {
    
}

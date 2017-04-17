use std::fmt;
use std::error::Error as StdError;
use std::result;

use matrix::Matrix;

#[derive(Debug)]
pub enum Error {
    WeightUpdateFailure,
    BiasUpdateFailure
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::WeightUpdateFailure => f.write_str("Error updating Weights in Layer"),
            Error::BiasUpdateFailure => f.write_str("Error updating Biases in Layer")
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Error::WeightUpdateFailure => "Error updating Weights in Layer",
            Error::BiasUpdateFailure => "Error updating Biases in Layer"
        }
    }
}

pub type LayerUpdateResult = result::Result<Matrix, Error>;

pub trait CombinatoryLayer {

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult;

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult;

}

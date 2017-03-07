use std::fmt;
use std::error::Error as StdError;
use std::result;

use matrix::Matrix;

#[derive(Debug)]
pub enum Error {
    ForwardPropFailure,
    BackPropFailure,
    WeightUpdateFailure,
    BiasUpdateFailure
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::ForwardPropFailure => f.write_str("Error Propagating Forward in Layer"),
            Error::BackPropFailure => f.write_str("Error Propagating Backward in Layer"),
            Error::WeightUpdateFailure => f.write_str("Error Updating Weights in Layer"),
            Error::BiasUpdateFailure => f.write_str("Error Updating Biases in Layer")
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Error::ForwardPropFailure => "Error Propagating Forward in Layer",
            Error::BackPropFailure => "Error Propagating Backward in Layer",
            Error::WeightUpdateFailure => "Error Updating Weights in Layer",
            Error::BiasUpdateFailure => "Error Updating Biases in Layer"
        }
    }
}

pub type ForwardPropResult = result::Result<Matrix, Error>;

pub type BackPropResult = result::Result<Matrix, Error>;

pub type WeightUpdateResult = result::Result<(), Error>;

pub type BiasUpdateResult = result::Result<(), Error>;

pub trait HiddenLayer {
    fn forward_prop(&mut self, input: &Matrix) -> ForwardPropResult;

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64) -> BackPropResult;

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix)  -> WeightUpdateResult;

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix) -> BiasUpdateResult;
}
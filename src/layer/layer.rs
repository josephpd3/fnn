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

/// An aliased type for a Result<Matrix, Error>
pub type ForwardPropResult = result::Result<Matrix, Error>;

/// An aliased type for a Result<Matrix, Error>
pub type BackPropResult = result::Result<Matrix, Error>;

/// An aliased type for a Result<Matrix, Error>
pub type WeightUpdateResult = result::Result<(), Error>;

/// An aliased type for a Result<Matrix, Error>
pub type BiasUpdateResult = result::Result<(), Error>;

pub trait Layer {
    /// Propagates input forward through the layer to produce an output Matrix
    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> ForwardPropResult;

    /// Propoagates a given derivative backward through the layer,
    /// updating the weights and biases within and producing a
    /// consecutive derivative Matrix to further propogate backwards
    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> BackPropResult;

    /// Given a learning rate and a Matrix of gradients for the weights,
    /// update the weights in the layer
    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize)  -> WeightUpdateResult;

    /// Given a learning rate and a Matrix of gradients for the biases,
    /// update the biases in the layer
    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> BiasUpdateResult;

    fn get_output_len(&self) -> usize;
}
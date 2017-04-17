use std::fmt;
use std::error::Error as StdError;
use std::result;

use matrix::Matrix;

#[derive(Debug)]
pub enum Error {
    ForwardPropFailure,
    BackPropFailure
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::ForwardPropFailure => f.write_str("Error Propagating Forward in Layer"),
            Error::BackPropFailure => f.write_str("Error Propagating Backward in Layer")
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Error::ForwardPropFailure => "Error Propagating Forward in Layer",
            Error::BackPropFailure => "Error Propagating Backward in Layer"
        }
    }
}

pub type PropagationResult = result::Result<Matrix, Error>;

pub trait BaseLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult;

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult;

    fn get_output_len(&self) -> usize;

}

use std::fmt;
use std::error::Error as StdError;
use std::result;

use matrix::Matrix;
use layer::base_layer::PropagationError;

// NEW ERROR HANDLING

#[derive(Debug)]
pub enum LayerUpdateError {
    Weights,
    Biases
}

impl fmt::Display for LayerUpdateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LayerUpdateError::Weights => f.write_str("Error updating Weights in Layer"),
            LayerUpdateError::Biases => f.write_str("Error updating Biases in Layer")
        }
    }
}

impl StdError for LayerUpdateError {
    fn description(&self) -> &str {
        match *self {
            LayerUpdateError::Weights => "Error updating Weights in Layer",
            LayerUpdateError::Biases => "Error updating Biases in Layer"
        }
    }
}

impl From<LayerUpdateError> for PropagationError {
    fn from(err: LayerUpdateError) -> PropagationError {
        PropagationError::Backward(Box::new(err))
    }
}

pub type LayerUpdateResult = result::Result<(), LayerUpdateError>;

pub trait CombinatoryLayer {

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult;

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult;

}

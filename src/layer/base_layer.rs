use std::fmt;
use std::error::Error as StdError;
use std::result;

use matrix::Matrix;

#[derive(Debug)]
pub enum PropagationError {
    Forward(Box<StdError>),
    Backward(Box<StdError>)
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PropagationError::Forward(ref err) => f.write_str(err.description()),
            PropagationError::Backward(ref err) => f.write_str(err.description())
        }
    }
}

impl StdError for PropagationError {
    fn description(&self) -> &str {
        match *self {
            PropagationError::Forward(ref err) => err.description(),
            PropagationError::Backward(ref err) => err.description()
        }
    }

    fn cause(&self) -> Option<&StdError> {
        match *self {
            PropagationError::Forward(ref err) => Some(err.as_ref()),
            PropagationError::Backward(ref err) => Some(err.as_ref())
        }
    }
}

/// Alias for a Result of propagation
pub type PropagationResult = result::Result<Matrix, PropagationError>;

/// Represents the base functionality of a Neural Network layer in that
/// representations are propagated forward and derivatives are propagated
/// backward.
pub trait BaseLayer {
    /// Given an input and a batch_size, performs the necessary actions of a given
    /// layer on forward propagation and returns the result within a PropagationResult
    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult;

    /// Given a derivative from the subsequent layer or output, performs actions
    /// necessary to calculate successive derivatives using the chain rule and
    /// returns the result within a PropagationResult
    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult;

    /// TODO: Determine if this can be depricated and just encapsulated as an updated
    /// member of a given layer's structure. Honestly, I rather like that the method
    /// exists here as a reminder to implement the functionality.
    fn get_output_len(&self) -> usize;

}

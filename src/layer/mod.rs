pub use self::layer::*;
pub use self::fully_connected::*;
pub use self::softmax::*;
pub use self::dropout::*;

// Newer layer abstractions
pub use self::base_layer::BaseLayer;
pub use self::combinatory_layer::CombinatoryLayer;
pub use self::activation_layer::ActivationLayer;

pub mod layer;
pub mod fully_connected;
pub mod softmax;
pub mod dropout;
pub mod dense;

// Newer layer abstractions
pub mod base_layer;
pub mod combinatory_layer;
pub mod activation_layer;


/// This enum allows the user to add layers to the network
/// by only passing in configuration parameters that can't be
/// abstracted into construction calculations like the number of
/// neurons in a layer or the number and size of filters in a
/// convolutional layer.
///
///
pub enum FNNLayer {
    FullyConnected {
        num_neurons: usize
    },
    Softmax {
        num_classes: usize
    },
    Dropout {
        rate: f64
    }
}

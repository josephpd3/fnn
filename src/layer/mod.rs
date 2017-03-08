pub use self::layer::*;
pub use self::fully_connected::*;
pub use self::softmax::*;

pub mod layer;
pub mod fully_connected;
pub mod softmax;


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
    }
}

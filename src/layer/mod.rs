pub use self::base_layer::BaseLayer;
pub use self::combinatory_layer::CombinatoryLayer;
pub use self::activation_layer::ActivationLayer;
pub use self::softmax::*;
pub use self::sigmoid::*;
pub use self::dense::*;
pub use self::dropout::*;

pub mod layer;
pub mod fully_connected;
pub mod softmax;
pub mod sigmoid;
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
/// DEPRECATED, MF!
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

pub enum Layer {
    Activation(Activation),
    Combination(Combination),
    Utility(Utility)
}

pub enum Activation {
    Sigmoid,
    Softmax
}

pub enum Combination {
    Dense {
        num_neurons: usize
    }
}

pub enum Utility {
    Dropout {
        rate: f64
    }
}

// ACTIVATIONS

#[macro_export]
macro_rules! sigmoid {
    () => {{
        Layer::Activation(Activation::Sigmoid)
    }};
}

#[macro_export]
macro_rules! softmax {
    () => {{
        Layer::Activation(Activation::Softmax)
    }};
}

// COMBINATIONS

#[macro_export]
macro_rules! dense {
    ($n_neurons:expr) => {{
        Layer::Combination(Combination::Dense{
            num_neurons: $n_neurons
        })
    }};
}

// UTILITIES

#[macro_export]
macro_rules! dropout {
    ($rate:expr) => {{
        Layer::Utility(Utility::Dropout{
            rate: $rate
        })
    }};
}

// #![allow(dead_code)] 
// enum Layer {
//     Activation(Activation),
//     Combination(Combination),
//     Dropout
// }

// enum Activation {
//     Sigmoid,
//     Softmax
// }

// enum Combination {
//     Dense,
//     Convolutional
// }


// fn main() {
//     let wat: Layer = Layer::Activation(Activation::Sigmoid);
//     match wat {
//         Layer::Activation(act) => match act {
//             Activation::Sigmoid => println!("Sigmoid!"),
//             Activation::Softmax => println!("Softmax!")
//         },
//         Layer::Combination(combo) => match combo {
//             Combination::Dense => println!("Dense!"),
//             Combination::Convolutional => println!("Convolutional!")
//         },
//         Layer::Dropout => println!("Dropout!")
//     }
// }

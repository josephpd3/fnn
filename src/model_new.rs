use dataset::DataSet;
use matrix::Matrix;

use loss::FNNLoss as Loss;
use optimizer::FNNOptimizer as Optimizer;

use layer::{
    base_layer::BaseLayer,
    combinatory_layer::CombinatoryLayer,
    activation_layer::ActivationLayer
};

use loss::{
    CrossEntropy
};

use optimizer::{
    Momentum,
    RMSProp
};

pub struct Model<T> {
    pub dataset: T,
    pub layers: Vec<Box<Layers>>,
    pub loss: Loss,
    pub optimizer: Optimizer,
}

impl<T> Model<T> where
    T: DataSet
{
    pub fn new(dataset: T, loss_function: Loss, optimizer: Optimizer) -> Model<T> {
        Model {
            dataset: dataset,
            layers: vec![],
            learning_rate: 0.1,
            loss: loss_function,
            optimizer: optimizer
        }
    }
}

use std::result;
use std::f64;

use dataset::DataSet;
use matrix::Matrix;

use loss::FNNLoss;
use optimizer::FNNOptimizer;

use layer::{
    BaseLayer,
    Layer,
    Activation,
    Combination,
    Utility,
    SoftmaxLayer,
    SigmoidLayer,
    DenseLayer,
    DropoutLayer
};
use layer::base_layer::PropagationError;

use loss::{
    Loss,
    CrossEntropy
};

use optimizer::{
    Optimizer,
    Momentum,
    NaiveMomentum,
    RMSProp
};

//#[macro_export]
// macro_rules! model {
//     (dataset:$ds:ident,
//      loss_function:$loss:item,
//      optimizer:$opt:item,
//      learning_rate:$lrate:item) => {{
//         Model::new(ModelDescription::New(
//             dataset: $ds,
//             loss_function: $loss,
//             optimizer: $opt,
//             learning_rate: $lrate
//         ))
//     }};
// }

pub struct Model<T> {
    pub dataset: T,
    pub layers: Vec<Box<BaseLayer>>,
    pub loss: Box<Loss>,
    pub optimizer: FNNOptimizer,
    pub learning_rate: f64
}

/// Enum providing a kwarg-like interface for initializing a Model
/// with more explicitly delineated arguments
pub enum ModelDescription<T> {
    New {
        dataset: T,
        loss_function: FNNLoss,
        optimizer: FNNOptimizer,
        learning_rate: f64
    }
}

pub type EpochResult = result::Result<f64, PropagationError>;

impl<T> Model<T> where
    T: DataSet
{
    pub fn new(desc: ModelDescription<T>) -> Model<T> {
        match desc {
            ModelDescription::New{
                dataset,
                loss_function,
                optimizer,
                learning_rate
            } => {
                let loss = match loss_function {
                    FNNLoss::CrossEntropy => {
                        Box::new(CrossEntropy::new())
                    }
                };

                Model {
                    dataset: dataset,
                    layers: vec![],
                    learning_rate: learning_rate,
                    loss: loss,
                    optimizer: optimizer
                }
            }
        }
    }

    fn get_optimizer_instance(&self) -> Box<Optimizer> {
        match self.optimizer {
            FNNOptimizer::Momentum{ mu } => {
                Box::new(Momentum::new(mu))
            },
            FNNOptimizer::NaiveMomentum{ mu } => {
                Box::new(NaiveMomentum::new(mu))
            }
            FNNOptimizer::RMSProp{ decay_rate, epsilon } => {
                Box::new(RMSProp::new(decay_rate, epsilon))
            }
        }
    }

    pub fn add(&mut self, layer: Layer) {
        let input_len: usize;

        match self.layers.last() {
            Some(boxed_layer) => {
                input_len = boxed_layer.as_ref().get_output_len();
            },
            None => {
                input_len = self.dataset.get_num_inputs();
            }
        }

        match layer {
            Layer::Activation(act) => match act {
                Activation::Sigmoid => {
                    self.layers.push(Box::new(SigmoidLayer::new(
                        input_len
                    )));
                },
                Activation::Softmax => {
                    self.layers.push(Box::new(SoftmaxLayer::new(
                        input_len
                    )));
                }
            },
            Layer::Combination(combo) => match combo {
                Combination::Dense{ num_neurons } => {
                    // Generate an optimizer to pass into the layer
                    let weight_opt = self.get_optimizer_instance();
                    let bias_opt = self.get_optimizer_instance();
                    self.layers.push(Box::new(DenseLayer::new(
                        input_len,
                        num_neurons,
                        weight_opt,
                        bias_opt
                    )));
                }
            },
            Layer::Utility(util) => match util {
                Utility::Dropout{ rate } => {
                    self.layers.push(Box::new(DropoutLayer::new(input_len, rate)));
                }
            }
        }
    }

    pub fn fit(&mut self, batch_size: usize, num_epochs: usize) {
        let mut last_loss = f64::NAN;

        for epoch in 0..num_epochs {
            println!("Starting Epoch {}!", epoch + 1);
            match self.run_epoch(batch_size) {
                Ok(loss) => {
                    if last_loss.is_nan() {
                        last_loss = loss;
                    } else if last_loss < loss {
                        self.learning_rate = self.learning_rate * 0.9;
                    }
                },
                _ => {}
            }
            self.dataset.replenish_minibatches();
        }
    }

    fn run_epoch(&mut self, batch_size: usize) -> EpochResult {
        let mut training_cases = 0;
        let total_training_cases = self.dataset.get_training_set_size();

        let mut last_output: Matrix;
        let mut last_deriv: Matrix;

        let mut loss: f64 = 0.0;
        let mut total_loss: f64 = 0.0;
        let mut avg_loss: f64 = 0.0;

        let mut is_training: bool = true;

        println!("Training...");
        while training_cases < total_training_cases {
            let minibatch = self.dataset.get_random_minibatch(batch_size);
            training_cases += minibatch.size;

            // "Input Layer"
            last_output = minibatch.input;

            // Iteration over all layers in forward order
            for layer in &mut self.layers {
                last_output = layer.forward_prop(&last_output, batch_size, is_training)?;
            }

            loss = self.loss.calculate_loss(&last_output, &minibatch.target, batch_size);
            total_loss += loss;
            avg_loss = total_loss / (training_cases / batch_size) as f64;

            if training_cases % 15_000 == 0 {
                println!("  Average accuracy at {} cases: {:.2}%\n    Loss Factor: {}", training_cases, (-avg_loss).exp() * 100.0, avg_loss);
            }

            last_deriv = self.loss.get_bp_deriv(&last_output, &minibatch.target);

            for layer in self.layers.iter_mut().rev() {
                last_deriv = layer.back_prop(&last_deriv, self.learning_rate, batch_size)?;
            }
        }

        is_training = false;

        println!("Running Validation for Epoch...");

        let validation_batch = self.dataset.get_validation_set();
        let mut last_validation_output = validation_batch.input;

        for layer in &mut self.layers {
            last_validation_output = layer.forward_prop(&last_validation_output, validation_batch.size, is_training)?;
        }

        let validation_loss = self.loss.calculate_loss(&last_validation_output, &validation_batch.target, validation_batch.size);

        println!("  Accuracy on Validation Set: {:.2}%", (-validation_loss).exp() * 100.0);
        Ok(validation_loss)
    }
}

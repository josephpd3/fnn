use std::convert::From;
use std::fmt;
use std::error::Error as StdError;
use std::result;

use layer::{
    Layer,
    FNNLayer,
    DropoutLayer,
    FullyConnectedLayer,
    SoftmaxLayer
};
use layer::Error as LayerError;
use dataset::DataSet;
use matrix::Matrix;

#[derive(Debug)]
/// Error representing a failure on the Model level
///
/// TODO: Implement [cause](https://doc.rust-lang.org/std/error/trait.Error.html) and ditch this lame `enum` format.
/// 
pub enum Error {
    LossFunctionFailure,
    EpochFailure,
    FittingFailure
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::LossFunctionFailure => f.write_str("Error Running Loss Function on Output"),
            Error::EpochFailure => f.write_str("Error Running Epoch: Epoch Fail!"),
            Error::FittingFailure => f.write_str("Error Fitting the Model")
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Error::LossFunctionFailure => "Error Running Loss Function on Output",
            Error::EpochFailure => "Error Running Epoch: Epoch Fail!",
            Error::FittingFailure => "Error Fitting the Model"
        }
    }
}

impl From<LayerError> for Error {
    fn from(err: LayerError) -> Error {
        Error::EpochFailure // Epic Fail!
    }
}

pub struct Model<T> {
    pub dataset: T,
    pub layers: Vec<Box<Layer>>,
    pub learning_rate: f64
}

pub type LossFunctionResult = result::Result<Matrix, Error>;
pub type EpochResult = result::Result<f64, Error>;
pub type FittingReult = result::Result<(), Error>;

#[allow(non_snake_case)] // For derivative variable names...
impl<T> Model<T> where
    T: DataSet
{
    pub fn new(dataset: T) -> Model<T> {
        Model {
            dataset: dataset,
            layers: vec![],
            learning_rate: 0.5
        }
    }

    pub fn add(&mut self, layer: FNNLayer) {
        let input_len: usize;

        match self.layers.last() {
            // If there are already layers in the model, base dimensionality of the
            // added layer on the output dimensionality of the last one
            Some(boxed_layer) => {
                input_len = boxed_layer.as_ref().get_output_len();
            },
            // ...otherwise, base the dimensionality of the added layer off of the
            // dimensionality of the network's input data
            None => {
                input_len = self.dataset.get_num_inputs();
            }
        }

        match layer {
            FNNLayer::FullyConnected{ num_neurons } => {
                self.layers.push(Box::new(FullyConnectedLayer::new(input_len, num_neurons)));
            },
            FNNLayer::Softmax{ num_classes } => {
                self.layers.push(Box::new(SoftmaxLayer::new(input_len, num_classes)));
            },
            FNNLayer::Dropout{ rate } => {
                self.layers.push(Box::new(DropoutLayer::new(input_len, rate)));
            }
        }
    }

    pub fn compile(&mut self) {
        // TODO
    }

    pub fn fit(&mut self, batch_size: usize, num_epochs: usize) -> FittingReult {
        let layer_output: Matrix;
        let batch_size = 100usize;
        let mut avg_CE: f64;
        let mut last_valid_CE: f64 = 1000.0;

        for epoch in 0..num_epochs {
            last_valid_CE = self.run_epoch(batch_size, last_valid_CE)?;
            println!("Done with Epoch {}!", epoch + 1usize);
        }

        Ok(())
    }

    fn validate(&mut self) {
        // TODO
    }

    fn test(&mut self) {
        // TODO
    }

    fn update_learning_rate(&mut self, avg_acc: f64, num_cases: usize, total_cases: usize) {
        let old_learning_rate = self.learning_rate;

        self.learning_rate = match (avg_acc * 100f64) as u8 {
            1...91 => self.learning_rate,
            91...100 => 0.45,
            _ => { self.learning_rate }
        };
    }

    fn run_epoch(&mut self, batch_size: usize, last_validation_CE: f64) -> EpochResult {
        let mut training_cases = 0usize;
        let total_training_cases = self.dataset.get_training_set_size();

        let mut last_output: Matrix;
        let mut last_deriv: Matrix;

        let mut cross_entropy: f64 = 0.0;
        let mut agg_cross_entropy: f64 = 0.0;
        let mut avg_cross_entropy: f64 = 0.0;

        let mut is_training: bool = true;

        // Run Epoch
        println!("Training...");
        while training_cases < total_training_cases {
            let minibatch = self.dataset.get_random_minibatch(batch_size);
            training_cases += minibatch.size;

            // Propogate Forward
            last_output = minibatch.input;
            for layer in &mut self.layers {
                last_output = layer.forward_prop(&last_output, batch_size, is_training)?;
            }

            // Calculate and Report Cross Entropy Loss for minibatch
            cross_entropy = self.calc_cross_entropy(&last_output, &minibatch.target, batch_size);
            agg_cross_entropy += cross_entropy;
            avg_cross_entropy = agg_cross_entropy / (training_cases / batch_size) as f64;

            if training_cases % 10_000 == 0 {
                println!("  Cross Entropy at {} cases: {:.5}; Average accuracy: {:.2}%", training_cases, avg_cross_entropy, (-avg_cross_entropy).exp() * 100f64);
            }

            // Propogate Backward
            last_deriv = &last_output - &minibatch.target;

            for layer in self.layers.iter_mut().rev() {
                last_deriv = layer.back_prop(&last_deriv, self.learning_rate, batch_size)?;
            }
        }

        is_training = false;

        // Validate Epoch
        println!("Running Validation for Epoch...");
        let validation_batch = self.dataset.get_validation_set();
        let mut last_validation_output = validation_batch.input;
        for layer in &mut self.layers {
            last_validation_output = layer.forward_prop(&last_validation_output, validation_batch.size, is_training)?;
        }
        let validation_CE = self.calc_cross_entropy(&last_validation_output, &validation_batch.target, validation_batch.size);
        println!("  Cross Entropy on Validation Set: {:.3}\n  Accuracy: {:.2}%", validation_CE, (-validation_CE).exp() * 100f64);

        if validation_CE > last_validation_CE {
            self.learning_rate = (self.learning_rate * 0.5).min(0.01);
        }

        // Replenish the stores!
        self.dataset.replenish_minibatches();

        Ok(validation_CE)
    }

    pub fn evaluate(&self) {
        // TODO
    }

    fn calc_cross_entropy(&self, output: &Matrix, target: &Matrix, batch_size: usize) -> f64 {
        // TODO: Make static method
        let log_output = output.mat_map(|x| x.ln());
        let unsummed = target.ew_multiply(&log_output); // Elementwise multiply
        let summed = unsummed.sum_rows().sum_cols();
        - (summed[0][0] / batch_size as f64)
    }

}

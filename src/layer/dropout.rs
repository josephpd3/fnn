use rand::StdRng;
use statrs::distribution::{Bernoulli, Distribution};

use layer::base_layer::{
    BaseLayer,
    PropagationResult
};
use matrix::Matrix;
use rand;

pub struct DropoutLayer {
    input_len: usize,
    output_len: usize,
    bern_coefficient: f64,
    bern: Bernoulli
}

impl DropoutLayer {

    pub fn new(input_len: usize, bern_coefficient: f64) -> Self {
        DropoutLayer {
            input_len: input_len,
            output_len: input_len,
            bern_coefficient: bern_coefficient,
            bern: Bernoulli::new(bern_coefficient).unwrap()
        }
    }

}

impl BaseLayer for DropoutLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult {
        // Seed a Random Number Generator
        let mut rng = rand::StdRng::new().unwrap();

        // Designate anonymous functions to use depending on whether the model
        // is training or testing and utilize them via a match statement
        
        let dropout = |x: f64| x * self.bern.sample::<StdRng>(&mut rng);
        // If all models are being combined in testing, scale values by the probability they will have been kept.
        let scale = |x: f64| x * self.bern_coefficient;

        match training {
            true => Ok(input.mat_map(dropout)),
            false => Ok(input.mat_map(scale))
        }
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult {
        // As the next layer recieves a zero for input in zeroed out value
        // locations, it will conveniently be backpropagated as a zero as well
        // when it is combined with the derivative in said subsequent layer
        Ok(bp_deriv.explicit_copy())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}
use rand::StdRng;
use statrs::distribution::{Bernoulli, Distribution};

use layer::layer::*;
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

impl Layer for DropoutLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> ForwardPropResult {
        let mut rng = rand::StdRng::new().unwrap();
        let dropout = |x: f64| x * self.bern.sample::<StdRng>(&mut rng);
        let scale = |x: f64| x * self.bern_coefficient;

        match training {
            true => Ok(input.mat_map(dropout)),
            false => Ok(input.mat_map(scale))
        }
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> BackPropResult {
        Ok(bp_deriv.explicit_copy())
    }

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> WeightUpdateResult {
        Ok(())
    }

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> BiasUpdateResult {
        Ok(())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}
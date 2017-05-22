use matrix::Matrix;
use layer::base_layer::{
    BaseLayer,
    PropagationResult
};
use layer::activation_layer::{
    ActivationLayer
};

pub struct SigmoidLayer {
    last_output: Option<Matrix>,
    input_len: usize,
    output_len: usize
}

impl SigmoidLayer {
    pub fn new(input_len: usize) -> Self {
        SigmoidLayer {
            last_output: None,
            input_len: input_len,
            output_len: input_len
        }
    }
}

#[allow(non_snake_case)] // For derivative variable names...
impl BaseLayer for SigmoidLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult {
        let output = input.mat_map(self.get_activation().as_ref());
        self.last_output = Some(output.explicit_copy());
        Ok(output)
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult {
        let last_output_ref = &self.last_output.take().unwrap();
        let dy_dz = last_output_ref.mat_map(self.get_derivative().as_ref());
        Ok(bp_deriv.ew_multiply(&dy_dz))
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}

impl ActivationLayer for SigmoidLayer {

    fn get_activation(&mut self) -> Box<Fn(f64) -> f64> {
        Box::new(|z: f64| -> f64 { 1.0 / (1.0 + (1.0 / z.exp())) })
    }

    fn get_derivative(&mut self) -> Box<Fn(f64) -> f64> {
        Box::new(|y: f64| -> f64 { y * (1.0 - y) })
    }

}

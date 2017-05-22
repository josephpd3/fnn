use matrix::Matrix;
use optimizer::Optimizer;
use layer::base_layer::{
    BaseLayer,
    PropagationResult
};
use layer::combinatory_layer::{
    CombinatoryLayer,
    LayerUpdateResult
};

/// A Neural Network layer modeled after classic hidden layer model
/// of a single dimensional row of neurons with a simple activation function
/// which, for now, is limited to the sigmoid function.
pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    weight_optimizer: Box<Optimizer>,
    bias_optimizer: Box<Optimizer>,
    last_input: Option<Matrix>,
    input_len: usize,
    output_len: usize
}

impl DenseLayer {

    pub fn new(input_len: usize, num_neurons: usize, weight_optimizer: Box<Optimizer>, bias_optimizer: Box<Optimizer>) -> Self {
        let initial_weights = Matrix::gaussian(input_len, num_neurons);
        let initial_biases = Matrix::gaussian(num_neurons, 1);

        DenseLayer {
            weights: initial_weights,
            biases: initial_biases,
            weight_optimizer: weight_optimizer,
            bias_optimizer: bias_optimizer,
            last_input: None,
            input_len: input_len,
            output_len: num_neurons
        }
    }

    fn get_batch_size_biases(&self, batch_size: usize) -> Matrix {
        let mut expanded_vec = vec![];
        for i in 0..self.biases.rows() {
            for _ in 0..batch_size {
                expanded_vec.push(self.biases[i][0]);
            }
        }
        Matrix::new(self.biases.rows(), batch_size, &expanded_vec)
    }

}

#[allow(non_snake_case)] // For derivative variable names...
impl BaseLayer for DenseLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult {
        self.last_input = Some(input.explicit_copy());

        let output = &(&self.weights.transpose() * input) + &self.get_batch_size_biases(batch_size);

        Ok(output)
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult {
        let last_input_ref = &self.last_input.take().unwrap();
        let dE_dz = bp_deriv;

        let dE_dw = last_input_ref * &dE_dz.transpose();
        let dE_dx = &self.weights * &dE_dz;

        self.update_weights(learning_rate, &dE_dw, batch_size)?;
        self.update_biases(learning_rate, &dE_dz.sum_cols(), batch_size)?;

        Ok(dE_dx)
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }
}

#[allow(non_snake_case)] // For derivative variable names...
impl CombinatoryLayer for DenseLayer {

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult {
        let scaled_gradient = gradient.mat_map(|x| x / batch_size as f64);
        let weights_delta = self.weight_optimizer.get_update(&self.weights, &scaled_gradient, learning_rate);
        self.weights = &self.weights + &weights_delta;
        self.weights = self.weights.restrict_col_norm(2.0);
        Ok(())
    }

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> LayerUpdateResult {
        let scaled_gradient = gradient.mat_map(|x| x / batch_size as f64);
        let bias_delta = self.bias_optimizer.get_update(&self.biases, &scaled_gradient, learning_rate);
        self.biases = &self.biases + &bias_delta;
        self.biases = self.biases.restrict_col_norm(2.0);
        Ok(())
    }

}

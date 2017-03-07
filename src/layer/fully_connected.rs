use layer::hidden_layer::*;
use matrix::Matrix;

struct FullyConnectedLayer {
    weights: Matrix,
    last_weight_update: Matrix,
    biases: Matrix,
    last_bias_update: Matrix,
    last_input: Matrix,
    last_output: Matrix,
    batch_size: usize
}

impl FullyConnectedLayer {

    fn new(input_len: usize, num_neurons: usize, batch_size: usize) -> Self {
        let initial_weights = Matrix::gaussian(num_neurons, input_len);
        let initial_biases = Matrix::gaussian(input_len, 1);

        FullyConnectedLayer {
            weights: initial_weights,
            last_weight_update: Matrix::zeroes(num_neurons, input_len),
            biases: initial_biases,
            last_bias_update: Matrix::zeroes(input_len, batch_size),
            last_input: Matrix::zeroes(input_len, batch_size),
            last_output: Matrix::zeroes(num_neurons, batch_size),
            batch_size: batch_size
        }
    }

    fn get_batch_size_biases(&self) -> Matrix {
        let mut expanded_vec = vec![];
        for i in 0..self.biases.rows() {
            for _ in 0..self.batch_size {
                expanded_vec.push(self.biases[i][0]);
            }
        }
        Matrix::new(self.biases.rows(), self.batch_size, &expanded_vec)
    }

}

#[allow(non_snake_case)] // For derivative variable names...
impl HiddenLayer for FullyConnectedLayer {

    fn forward_prop(&mut self, input: &Matrix) -> ForwardPropResult {
        let sigmoid = |z: f64| 1f64 / (1f64 + (1f64 / z.exp()));

        let to_activate = &(&self.weights * input) + &self.get_batch_size_biases();
        Ok(to_activate.mat_map(sigmoid))
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64) -> BackPropResult {
        let dy_dz = self.last_output.ew_multiply(&self.last_output.mat_map(|y| 1f64 - y));
        let dE_dz = bp_deriv.ew_multiply(&dy_dz);
        let dE_dw = &self.last_input * &dE_dz.transpose();
        let dE_dx = &self.weights * &bp_deriv;

        self.update_weights(learning_rate, &dE_dw)?;
        self.update_biases(learning_rate, &dE_dz.sum_cols())?;

        Ok(dE_dx)
    }

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix)  -> WeightUpdateResult {
        let weights_delta = &self.last_weight_update + gradient;
        self.weights = &self.weights - &(weights_delta.mat_map(|x| x * learning_rate));
        self.last_weight_update = weights_delta;
        Ok(())
    }

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix) -> BiasUpdateResult {
        let bias_delta = &self.last_bias_update + gradient;
        self.biases = &self.biases - &(bias_delta.mat_map(|x| x * learning_rate));
        self.last_bias_update = bias_delta;
        Ok(())
    }
}
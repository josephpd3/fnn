use layer::layer::*;
use matrix::Matrix;

/// A Neural Network layer modeled after classic hidden layer model
/// of a single dimensional row of neurons with a simple activation function
/// which, for now, is limited to the sigmoid function.
pub struct FullyConnectedLayer {
    weights: Matrix,
    last_weight_update: Option<Matrix>,
    biases: Matrix,
    last_bias_update: Option<Matrix>,
    last_input: Option<Matrix>,
    last_output: Option<Matrix>,
    input_len: usize,
    output_len: usize
}

impl FullyConnectedLayer {

    pub fn new(input_len: usize, num_neurons: usize) -> Self {
        let initial_weights = Matrix::gaussian(input_len, num_neurons);
        let initial_biases = Matrix::gaussian(num_neurons, 1);

        FullyConnectedLayer {
            weights: initial_weights,
            last_weight_update: None,
            biases: initial_biases,
            last_bias_update: None,
            last_input: None,
            last_output: None,
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
impl Layer for FullyConnectedLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize) -> ForwardPropResult {
        let sigmoid = |z: f64| 1f64 / (1f64 + (1f64 / z.exp()));

        self.last_input = Some(input.explicit_copy()); // Store most recent input for backprop

        let to_activate = &(&self.weights.transpose() * input) + &self.get_batch_size_biases(batch_size);
        let output = to_activate.mat_map(sigmoid);

        self.last_output = Some(output.explicit_copy()); // Store most recent output for backprop

        Ok(output)
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> BackPropResult {
        // Consume the existing matrices inside the Option<> and replace them with None.
        // Rightuflly panics if a forward_prop wasn't performed, saving the last input
        // and output
        let last_input_ref = &self.last_input.take().unwrap();
        let last_output_ref = &self.last_output.take().unwrap();

        let dy_dz = last_output_ref.ew_multiply(&last_output_ref.mat_map(|y| 1.0f64 - y));
        let dE_dz = bp_deriv.ew_multiply(&dy_dz);
        let dE_dw = last_input_ref * &dE_dz.transpose();
        let dE_dx = &self.weights * &dE_dz;

        self.update_weights(learning_rate, &dE_dw, batch_size)?;
        self.update_biases(learning_rate, &dE_dz.sum_cols(), batch_size)?;

        Ok(dE_dx)
    }

    fn update_weights(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> WeightUpdateResult {
        let weights_delta: Matrix;
        let momentum = 0.9f64;

        // Consumes the last update if it took place
        match self.last_weight_update.take() {
            Some(update_matrix) => {
                weights_delta = &update_matrix.mat_map(|x| x * momentum) + &gradient.mat_map(|x| x / batch_size as f64);
            },
            None => {
                weights_delta = gradient.mat_map(|x| x / batch_size as f64);
            }
        }

        self.weights = &self.weights - &(weights_delta.mat_map(|x| x * learning_rate));
        self.last_weight_update = Some(weights_delta);
        Ok(())
    }

    fn update_biases(&mut self, learning_rate: f64, gradient: &Matrix, batch_size: usize) -> BiasUpdateResult {
        let bias_delta: Matrix;
        let momentum = 0.9f64;

        // Consumes the last update if it took place
        match self.last_bias_update.take() {
            Some(update_matrix) => {
                bias_delta = &update_matrix.mat_map(|x| x * momentum) + &gradient.mat_map(|x| x / batch_size as f64);
            },
            None => {
                bias_delta = gradient.mat_map(|x| x / batch_size as f64);
            }
        }

        self.biases = &self.biases - &(bias_delta.mat_map(|x| x * learning_rate));
        self.last_bias_update = Some(bias_delta);
        Ok(())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }
}

use layer::layer::*;
use matrix::Matrix;
use std::cmp;

pub struct SoftmaxLayer {
    // Weights
    pub weights: Matrix,
    weight_learning_rates: Matrix,
    weight_theta: f64,
    weight_kappa: f64,
    weight_phi: f64,
    weight_gradient_exponential_averages: Matrix,
    last_weight_update: Option<Matrix>,
    // Biases
    pub biases: Matrix,
    bias_theta: f64,
    bias_kappa: f64,
    bias_phi: f64,
    bias_learning_rates: Matrix,
    bias_gradient_exponential_averages: Matrix,
    last_bias_update: Option<Matrix>,
    // Storage
    last_input: Option<Matrix>,
    last_output: Option<Matrix>,
    // Dimensionality
    pub input_len: usize,
    pub output_len: usize
}

impl SoftmaxLayer {

    pub fn new(input_len: usize, num_classes: usize) -> Self {
        // Set up weight and update storage
        let initial_weights = Matrix::gaussian(input_len, num_classes);
        let initial_weight_learning_rates = Matrix {
            rows: input_len,
            cols: num_classes,
            data: vec![vec![0.8f64; num_classes]; input_len]
        };
        let initial_weight_gradient_exponential_averages = Matrix::zeroes(input_len, num_classes);

        // Set up bias and update storage
        let initial_biases = Matrix::gaussian(num_classes, 1);
        let initial_bias_learning_rates = Matrix {
            rows: num_classes,
            cols: 1usize,
            data: vec![vec![0.8f64; num_classes]; input_len]
        };
        let initial_bias_gradient_exponential_averages = Matrix::zeroes(num_classes, 1);

        SoftmaxLayer {
            // Weights
            weights: initial_weights,
            weight_learning_rates: initial_weight_learning_rates,
            weight_theta: 0.9f64,
            weight_kappa: 0.1f64,
            weight_phi: 0.5f64,
            weight_gradient_exponential_averages: initial_weight_gradient_exponential_averages,
            last_weight_update: None,
            // Biases
            biases: initial_biases,
            bias_learning_rates: initial_bias_learning_rates,
            bias_theta: 0.9f64,
            bias_kappa: 0.1f64,
            bias_phi: 0.5f64,
            bias_gradient_exponential_averages: initial_bias_gradient_exponential_averages,
            last_bias_update: None,
            // Storage
            last_input: None,
            last_output: None,
            // Dimensionality
            input_len: input_len,
            output_len: num_classes
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
impl Layer for SoftmaxLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize) -> ForwardPropResult {
        self.last_input = Some(input.explicit_copy());

        let to_activate = &(&self.weights.transpose() * input) + &self.get_batch_size_biases(batch_size);
        let exponentiated = to_activate.mat_map(|x| x.exp());
        let summed = exponentiated.sum_rows();

        let mut activated_outer = vec![];

        for row in 0..exponentiated.rows {
            let mut activated_inner = vec![];
            for col in 0..exponentiated.cols {
                activated_inner.push(exponentiated[row][col] / summed[0][col]);
            }
            activated_outer.push(activated_inner);
        }

        let output = Matrix {
            rows: self.output_len,
            cols: batch_size,
            data: activated_outer
        };

        self.last_output = Some(output.explicit_copy());

        Ok(output)
    }

    fn back_prop(&mut self, bp_deriv: &Matrix, batch_size: usize) -> BackPropResult {
        // The loss function actually takes us all the way to dC_dz!
        let last_input_ref = &self.last_input.take().unwrap();
        let last_output_ref = &self.last_output.take().unwrap();

        let dE_dw = last_input_ref * &bp_deriv.transpose();
        let dE_dx = &self.weights * bp_deriv;

        self.update_weights(&dE_dw, batch_size)?;
        self.update_biases(&bp_deriv.sum_cols(), batch_size)?;

        Ok(dE_dx)
    }

    fn update_weights(&mut self, gradient: &Matrix, batch_size: usize) -> WeightUpdateResult {
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

        self.update_weight_learning_rates(&gradient);

        let limit_delta = |d: f64 | -> f64 {
            match d {
                val if val > 0.0 => val.min(50.0).max(10.0e-7),
                val if val > 0.0 => val.max(-50.0).min(-10.0e-7),
                _ => d
            }
        };

        self.weights = &self.weights - &weights_delta.ew_multiply(&self.weight_learning_rates)
                                                     .mat_map(limit_delta);
        self.last_weight_update = Some(weights_delta);
        Ok(())
    }

    fn update_weight_learning_rates(&mut self, weight_gradient: &Matrix) -> WeightLearningRateUpdateResult {
        let mut current_trend: f64;
        let mut outer = vec![];

        for row in 0..self.weight_learning_rates.rows {
            let mut inner = vec![];

            for col in 0..self.weight_learning_rates.cols {
                let learning_rate_delta: f64;

                current_trend = weight_gradient[row][col] * self.weight_gradient_exponential_averages[row][col];

                if current_trend > 0.0 {
                    learning_rate_delta = self.weight_kappa;
                } else if current_trend < 0.0 {
                    learning_rate_delta = -self.weight_phi * self.weight_learning_rates[row][col];
                } else {
                    learning_rate_delta = 0.0;
                }

                inner.push(learning_rate_delta);
            }
            outer.push(inner);
        }

        let learning_rate_delta_mat = Matrix {
            rows: self.weight_learning_rates.rows,
            cols: self.weight_learning_rates.cols,
            data: outer
        };

        self.weight_learning_rates = &self.weight_learning_rates + &learning_rate_delta_mat;
        self.weight_gradient_exponential_averages = weight_gradient.mat_map(|dw| (1.0 - self.weight_theta) * dw) + self.weight_gradient_exponential_averages.mat_map(|a| a * self.weight_theta);

        Ok(())
    }

    fn update_biases(&mut self, gradient: &Matrix, batch_size: usize) -> BiasUpdateResult {
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

        self.update_bias_learning_rates(&gradient);

        self.biases = &self.biases - &bias_delta.ew_multiply(&self.bias_learning_rates);
        self.last_bias_update = Some(bias_delta);
        Ok(())
    }

    fn update_bias_learning_rates(&mut self, bias_gradient: &Matrix) -> BiasLearningRateUpdateResult {
        let mut current_trend: f64;
        let mut outer = vec![];

        for row in 0..self.bias_learning_rates.rows {
            let mut inner = vec![];

            for col in 0..self.bias_learning_rates.cols {
                let learning_rate_delta: f64;

                current_trend = bias_gradient[row][col] * self.bias_gradient_exponential_averages[row][col];

                if current_trend > 0.0 {
                    learning_rate_delta = self.bias_kappa;
                } else if current_trend < 0.0 {
                    learning_rate_delta = -self.bias_phi * self.bias_learning_rates[row][col];
                } else {
                    learning_rate_delta = 0.0;
                }

                inner.push(learning_rate_delta);
            }
            outer.push(inner);
        }

        let learning_rate_delta_mat = Matrix {
            rows: self.bias_learning_rates.rows,
            cols: self.bias_learning_rates.cols,
            data: outer
        };

        self.bias_learning_rates = &self.bias_learning_rates + &learning_rate_delta_mat;
        self.bias_gradient_exponential_averages = bias_gradient.mat_map(|dx| (1.0 - self.bias_theta) * dx) + self.bias_gradient_exponential_averages.mat_map(|a| a * self.bias_theta);

        Ok(())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}

use layer::layer::*;
use matrix::Matrix;

pub struct SoftmaxLayer {
    pub weights: Matrix,
    last_weight_update: Option<Matrix>,
    pub biases: Matrix,
    last_bias_update: Option<Matrix>,
    last_input: Option<Matrix>,
    last_output: Option<Matrix>,
    pub input_len: usize,
    pub output_len: usize
}

impl SoftmaxLayer {

    pub fn new(input_len: usize, num_classes: usize) -> Self {
        let initial_weights = Matrix::gaussian(input_len, num_classes);
        let initial_biases = Matrix::gaussian(num_classes, 1);

        SoftmaxLayer {
            weights: initial_weights,
            last_weight_update: None,
            biases: initial_biases,
            last_bias_update: None,
            last_input: None,
            last_output: None,
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

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> ForwardPropResult {
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

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> BackPropResult {
        // The loss function actually takes us all the way to dC_dz!
        let last_input_ref = &self.last_input.take().unwrap();
        let last_output_ref = &self.last_output.take().unwrap();

        let dE_dw = last_input_ref * &bp_deriv.transpose();
        let dE_dx = &self.weights * bp_deriv;

        self.update_weights(learning_rate, &dE_dw, batch_size)?;
        self.update_biases(learning_rate, &bp_deriv.sum_cols(), batch_size)?;

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
        //self.weights = self.weights.restrict_col_norm(2.0);
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
        //self.biases = self.biases.restrict_col_norm(2.0);
        self.last_bias_update = Some(bias_delta);
        Ok(())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}

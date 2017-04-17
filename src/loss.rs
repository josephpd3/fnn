use matrix::Matrix;

pub trait Loss {
    fn calculate_loss(&self, output: &Matrix, target: &Matrix, batch_size: usize) -> f64;

    fn get_bp_deriv(&self, output: &Matrix, target: &Matrix) -> Matrix;

    fn get_name(&self) -> &'static str;
}

pub enum FNNLoss {
    CrossEntropy
}

pub struct CrossEntropy {
    name: &'static str
}

impl CrossEntropy {
    fn new() -> Self {
        CrossEntropy {
            name: "Categorical Cross Entropy"
        }
    }
}

impl Loss for CrossEntropy {
    fn calculate_loss(&self, output: &Matrix, target: &Matrix, batch_size: usize) -> f64 {
        let log_output = output.mat_map(|x| x.ln());
        let unsummed = target.ew_multiply(&log_output); // Elementwise multiply
        let summed = unsummed.sum_rows().sum_cols();
        - (summed[0][0] / batch_size as f64)
    }

    fn get_bp_deriv(&self, output: &Matrix, target: &Matrix) -> Matrix {
        target.ew_multiply(&output.mat_map(|x| -x))
    }

    fn get_name(&self) -> &'static str {
        self.name
    }
}

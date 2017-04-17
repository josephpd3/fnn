use matrix::Matrix;

pub trait Loss {
    fn calculate_loss(&self, output: &Matrix, target: &Matrix, batch_size: usize) -> f64;

    fn get_bp_deriv(&self, output: &Matrix, target: &Matrix) -> Matrix;
}

pub struct CrossEntropy { }

impl CrossEntropy {
    fn new() -> Self {
        CrossEntropy {}
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
}

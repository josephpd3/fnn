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
    pub fn new() -> Self {
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
        target.ew_multiply(&output.mat_map(|x| -(1.0 / x)))
    }

    fn get_name(&self) -> &'static str {
        self.name
    }
}

#[test]
fn calculate_cross_entropy_deriv() {
    // Utility function to determine equivalence of matrices given tolerance
    fn matrices_eq_within_tol(m1: &Matrix, m2: &Matrix, tol: f64) -> bool {
        for row in 0..m1.rows {
            for col in 0..m2.cols {
                if (m1[row][col] - m2[row][col]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    let mut ce = CrossEntropy::new();
    let output = Matrix::new(3, 3, &vec![0.343f64, 0.247f64, 0.258f64,
                                         0.409f64, 0.302f64, 0.316f64,
                                         0.248f64, 0.451f64, 0.426f64]);
    let target = Matrix::new(3, 3, &vec![0.0f64, 0.0f64, 1.0f64,
                                         1.0f64, 1.0f64, 0.0f64,
                                         0.0f64, 0.0f64, 0.0f64]);
    let dL_dy = Matrix::new(3, 3, &vec![0.000f64, 0.000f64, -3.876f64,
                                        -2.445f64, -3.311f64, 0.000f64,
                                        0.000f64, 0.000f64, 0.000f64]);

    let deriv = ce.get_bp_deriv(&output, &target);

    assert!(matrices_eq_within_tol(
        &deriv,
        &dL_dy,
        1.0e-2f64
    ), "BP_Deriv Miscalculated!");
}

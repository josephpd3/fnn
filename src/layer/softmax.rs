//use rayon::prelude::*;

use matrix::Matrix;
use layer::base_layer::{
    BaseLayer,
    PropagationResult
};
use layer::activation_layer::{
    ActivationLayer
};

pub struct SoftmaxLayer {
    last_output: Option<Matrix>,
    input_len: usize,
    output_len: usize
}

impl SoftmaxLayer {
    pub fn new(input_len: usize) -> Self {
        SoftmaxLayer {
            last_output: None,
            input_len: input_len,
            output_len: input_len
        }
    }
}

impl BaseLayer for SoftmaxLayer {

    fn forward_prop(&mut self, input: &Matrix, batch_size: usize, training: bool) -> PropagationResult {
        let exponentiated = input.mat_map(|x| x.exp());
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

    fn back_prop(&mut self, bp_deriv: &Matrix, learning_rate: f64, batch_size: usize) -> PropagationResult {
        let last_output_ref = &self.last_output.take().unwrap();
        let deriv_transpose = bp_deriv.transpose();

        let data = last_output_ref.transpose().data
            .iter()
            .enumerate()
            .map(|(case_idx, case)| {

                let mut softmax_comps = vec![];

                // Pre-compute all potential dy_dz terms,
                // but only the lower triangle of the matrix,
                // as the upper and lower triangles are,
                // matehmatically, reflections.
                for i in 0..case.len() {

                    let mut softmax_comps_inner = vec![];

                    for j in 0..(i + 1) {
                        let factor: f64;
                        if i == j {
                            factor = case[i] * (1.0 - case[i]);
                        } else {
                            factor = - case[i] * case[j];
                        }
                        softmax_comps_inner.push(factor);
                    }

                    softmax_comps.push(softmax_comps_inner);
                }

                let case_derivs = &deriv_transpose[case_idx];

                (0..case.len()).map(|deriv_idx| {
                    let mut sum = 0.0;
                    for output_idx in 0..case.len() {
                        // Depending on whether i == j,
                        // multiply derivitive of Loss (wrt output)
                        // by derivitive of output (wrt input)
                        // wrt whether j also represents the dominant term
                        sum += case_derivs[output_idx] * {
                            if deriv_idx > output_idx {
                                softmax_comps[deriv_idx][output_idx]
                            } else {
                                softmax_comps[output_idx][deriv_idx]
                            }
                        };
                    }
                    sum
                }).collect::<Vec<f64>>()

            }).collect::<Vec<Vec<f64>>>();

        let output = Matrix {
            rows: bp_deriv.cols,
            cols: bp_deriv.rows,
            data: data
        };

        Ok(output.transpose())
    }

    fn get_output_len(&self) -> usize {
        self.output_len
    }

}

impl ActivationLayer for SoftmaxLayer {

    fn get_activation(&mut self) -> Box<Fn(f64) -> f64> {
        Box::new(|x: f64| x)
    }

    fn get_derivative(&mut self) -> Box<Fn(f64) -> f64> {
        Box::new(|x: f64| x)
    }

}

#[test]
fn forward_and_backward() {
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

    let mut l = SoftmaxLayer::new(3);
    let z = Matrix::new(3, 3, &vec![0.5f64, 0.3f64, 0.2f64,
                                    0.7f64, 0.5f64, 0.4f64,
                                    0.2f64, 0.9f64, 0.7f64]);
    let batch_size = 3usize;
    let y_goal = Matrix::new(3, 3, &vec![0.343f64, 0.247f64, 0.258f64,
                                         0.409f64, 0.302f64, 0.316f64,
                                         0.248f64, 0.451f64, 0.426f64]);
    let forward = l.forward_prop(&z, batch_size, false);

    match forward {
        Ok(y_mat) => assert!(matrices_eq_within_tol(
            &y_mat,
            &y_goal,
            1.0e-2f64
        ), "Forward Propagation Miscalculated!"),
        _ => panic!("Couldn't propagate forward!")
    }

    // Precalculated Cross entropy loss wrt target data
    // [0, 0, 1,
    //  1, 1, 0,
    //  0, 0, 0]
    let dL_dy = Matrix::new(3, 3, &vec![0.000f64, 0.000f64, -3.876f64,
                                        -2.445f64, -3.311f64, 0.000f64,
                                        0.000f64, 0.000f64, 0.000f64]);
    let bp_goal = Matrix::new(3, 3, &vec![0.343f64, 0.247f64, -0.742f64,
                                          -0.591f64, -0.698f64, 0.316f64,
                                          0.248f64, 0.451f64, 0.426f64]);

    let backward = l.back_prop(&dL_dy, 0.0, batch_size);

    match backward {
        Ok(bp_mat) => assert!(matrices_eq_within_tol(
            &bp_mat,
            &bp_goal,
            1.0e-2f64
        ), "Backward Propagation Miscalculated!"),
        _ => panic!("Couldn't propagate backward!")
    }
}

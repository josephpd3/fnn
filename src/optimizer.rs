use matrix::Matrix;

pub trait Optimizer {
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix;
}

pub enum FNNOptimizer {
    Momentum {
        mu: f64
    },
    RMSProp {
        decay_rate: f64
    }
}

pub struct Momentum {
    mu: f64,
    velocity: Option<Matrix>
}

impl Momentum {
    fn new(mu: f64) -> Self {
        Momentum {
            mu: mu,
            velocity: None
        }
    }
}

impl Optimizer for Momentum {
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix {
        let updated_v: Matrix;

        match self.velocity.take() {
            Some(velocity) => {
                updated_v = &velocity.mat_map(|x| self.mu * x) - &deriv.mat_map(|x| learning_rate * x);
            },
            None => {
                let temp_v = Matrix::zeroes(parameters.rows, parameters.cols);
                updated_v = &temp_v.mat_map(|x| self.mu * x) - &deriv.mat_map(|x| learning_rate * x);
            }
        }

        self.velocity = Some(updated_v.explicit_copy());
        updated_v
    }
}

pub struct RMSProp {
    decay_rate: f64,
    epsilon: f64,
    cache: Option<Matrix>
}

impl RMSProp {
    fn new(decay_rate: f64, epsilon: f64) -> Self {
        RMSProp {
            decay_rate: decay_rate,
            epsilon: epsilon,
            cache: None
        }
    }
}

impl Optimizer for RMSProp {
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix {
        let updated_cache: Matrix;

        match self.cache.take() {
            Some(cache) => {
                updated_cache = &cache.mat_map(|x| self.decay_rate * x) + &deriv.mat_map(|x| (1.0 - self.decay_rate) * x.powi(2));
            },
            None => {
                let temp_cache = Matrix::zeroes(parameters.rows, parameters.cols);
                updated_cache = &temp_cache.mat_map(|x| self.decay_rate * x) + &deriv.mat_map(|x| (1.0 - self.decay_rate) * x.powi(2));
            }
        }

        self.cache = Some(updated_cache.explicit_copy());
        deriv.ew_multiply(&updated_cache.mat_map(|x| x.sqrt() ))
    }
}

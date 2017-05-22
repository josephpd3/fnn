use matrix::Matrix;

/// This trait enables the abstraction of Optimization algorithms--such as
/// Momentum, Nesterov Momentum, RMSProp, and Adadelta--for use in a
/// Model built with Ferrous Neural Networks. The advantage here is that things
/// such as hyperparameters, updating caches and auxilliary variables, and other
/// factors involved in the process can be encapsulated within a structure that
/// implements this trait.
pub trait Optimizer {
    /// Takes network parameters, a derivative wrt said parameters, and a
    /// learning rate to calculate an update based on the particular
    /// Optimization method given.
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix;
}

/// This enumeration provides an interface with which an API consumer can pass
/// instructions for constructing a desired Optimizer into the Model so that it
/// knows what kind of Optimizer to inject into each layer that requires one.
pub enum FNNOptimizer {
    /// Traditional Momentum
    Momentum {
        mu: f64
    },
    /// My momentum implementation from older versions of FNN
    NaiveMomentum {
        mu: f64
    },
    /// Root-Mean-Square Prop as made popular by Geoffrey Hinton's Coursera course
    RMSProp {
        decay_rate: f64,
        epsilon: f64
    }
}

/// This struct serves as the basis for traditional Momentum in a Neural Network
/// which preserves the direction of previous derviatives and utilizes it to
/// minimize the shifting (+/-) nature of constantly changing gradients.
pub struct Momentum {
    /// momentum scaling factor
    mu: f64,
    /// storage for accumulated velocity
    velocity: Option<Matrix>
}

impl Momentum {
    pub fn new(mu: f64) -> Self {
        Momentum {
            mu: mu,
            velocity: None
        }
    }
}

impl Optimizer for Momentum {
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix {
        let updated_v: Matrix;

        // Either use an existing cache or instantiate a basis of zeroes
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

/// This struct replicates Momentum as I first implemented it in older versions
pub struct NaiveMomentum {
    mu: f64,
    velocity: Option<Matrix>
}

impl NaiveMomentum {
    pub fn new(mu: f64) -> Self {
        NaiveMomentum {
            mu: mu,
            velocity: None
        }
    }
}

impl Optimizer for NaiveMomentum {
    fn get_update(&mut self, parameters: &Matrix, deriv: &Matrix, learning_rate: f64) -> Matrix {
        let updated_v: Matrix;

        match self.velocity.take() {
            Some(velocity) => {
                updated_v = &velocity.mat_map(|x| self.mu * x) + &deriv;
            },
            None => {
                updated_v = deriv.explicit_copy();
            }
        }

        let scaled_v = updated_v.mat_map(|x| x * learning_rate);

        self.velocity = Some(updated_v.explicit_copy());
        scaled_v.mat_map(|x| -x)
    }
}

/// This struct serves as the basis for RMSProp as popularized by Geoffrey Hinton's
/// Coursera course on Neural Networks. RMSProp utilizes the magnitude of previous
/// gradients to regulate how much the learning rate and current gradient impact
/// updates to the parameters of the network.
pub struct RMSProp {
    decay_rate: f64,
    epsilon: f64,
    cache: Option<Matrix>
}

impl RMSProp {
    pub fn new(decay_rate: f64, epsilon: f64) -> Self {
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

        // Either use an existing cache or instantiate a basis of zeroes
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

        // x.abs() is necessary prior to calculation of .sqrt() as Rust, when
        // raising a negative f64 value to a power between 0 and 1, returns NAN
        deriv.ew_multiply(&updated_cache.mat_map(|x| (x.abs().sqrt() + self.epsilon).powi(-1))).mat_map(|x| -learning_rate * x)
    }
}

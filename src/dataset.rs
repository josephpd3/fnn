use matrix::Matrix;

// TODO: Be damn sure to work towards making the consumption of a dataset
// more idiomatic. This is disgusting af.

/// This encapsulates data to be passed into the Model to be learned with.
pub struct Batch {
    pub size: usize,
    pub input: Matrix,
    pub target: Matrix
}

/// This trait allows any dataset to be passed into the Model as long as it
/// fulfills the basic, required functions
pub trait DataSet {
    fn get_random_minibatch(&self, batch_size: usize) -> Batch;

    fn get_training_set_size(&self) -> usize;

    fn get_validation_set(&self) -> Batch;

    fn get_test_set(&self) -> Batch;

    fn get_num_inputs(&self) -> usize;
}
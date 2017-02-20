use ndarray::prelude::*;
use ndarray::Data;

pub struct DataSet<S, D>
    where S: Data
{
    data: ArrayBase<S, D>
}

pub struct TrainingData<S, D> 
    where S: Data
{
    train: DataSet<S, D>,
    valid: DataSet<S, D>,
    test: DataSet<S, D>
}
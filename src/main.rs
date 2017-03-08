extern crate csv;
extern crate rand;
extern crate rustc_serialize;

#[macro_use(matrix)]
extern crate fnn;

use csv::Reader;
use fnn::matrix::Matrix;
use fnn::dataset::Batch;

use fnn::FNNLayer as Layer;
use fnn::Model;
use fnn::DataSet;

#[derive(RustcDecodable)]
struct MNISTRecord {
    class: u8,
    image_data: Vec<u8>,
}

pub struct MNISTDataSet {
    training_x: Matrix,
    training_y: Matrix,
    batch_bank_x: Matrix,
    batch_bank_y: Matrix,
    validation_x: Matrix,
    validation_y: Matrix,
    test_x: Matrix,
    test_y: Matrix,
    pub training_set_size: usize,
    pub validation_set_size: usize,
    pub test_set_size: usize,
    pub num_inputs: usize,
    pub num_classes: usize
}

impl MNISTDataSet {
    fn new(records: Vec<MNISTRecord>,
           training_set_size: usize,
           validation_set_size: usize,
           test_set_size: usize) -> Self {
        // Break down records into matrices for use
        // 1) Break down training set
        let mut training_x_outer = vec![];
        let mut training_y_outer = vec![];

        for idx in 0..training_set_size {
            training_x_outer.push(records[idx].image_data.iter().map(|x| *x as f64).collect::<Vec<f64>>());
            training_y_outer.push(MNISTDataSet::create_class_onehot(records[idx].class));
        }
        // 2) Break down validation set
        let mut validation_x_outer = vec![];
        let mut validation_y_outer = vec![];

        for idx in training_set_size..(training_set_size + validation_set_size) {
            validation_x_outer.push(records[idx].image_data.iter().map(|x| *x as f64).collect::<Vec<f64>>());
            validation_y_outer.push(MNISTDataSet::create_class_onehot(records[idx].class));
        }

        // 3) Break down test set
        let mut test_x_outer = vec![];
        let mut test_y_outer = vec![];

        for idx in validation_set_size..(validation_set_size + test_set_size) {
            test_x_outer.push(records[idx].image_data.iter().map(|x| *x as f64).collect::<Vec<f64>>());
            test_y_outer.push(MNISTDataSet::create_class_onehot(records[idx].class));
        }

        let case_dim = 784usize;
        let class_dim = 10usize;

        let tr_x = Matrix {
            rows: training_set_size,
            cols: case_dim,
            data: training_x_outer
        };
        let tr_y = Matrix {
            rows: training_set_size,
            cols: class_dim,
            data: training_y_outer
        };

        let v_x = Matrix {
            rows: validation_set_size,
            cols: case_dim,
            data: validation_x_outer
        };
        let v_y = Matrix {
            rows: validation_set_size,
            cols: class_dim,
            data: validation_y_outer
        };

        let ts_x = Matrix {
            rows: test_set_size,
            cols: case_dim,
            data: test_x_outer
        };
        let ts_y = Matrix {
            rows: test_set_size,
            cols: class_dim,
            data: test_y_outer
        };

        // Send her away!
        MNISTDataSet {
            training_x: tr_x.explicit_copy(),
            training_y: tr_y.explicit_copy(),
            batch_bank_x: tr_x,
            batch_bank_y: tr_y,
            validation_x: v_x.transpose(),
            validation_y: v_y.transpose(),
            test_x: ts_x.transpose(),
            test_y: ts_y.transpose(),
            training_set_size: training_set_size,
            validation_set_size: validation_set_size,
            test_set_size: test_set_size,
            num_inputs: case_dim,
            num_classes: class_dim
        }
    }

    fn create_class_onehot(class: u8) -> Vec<f64> {
        let mut one_hot_vec = vec![0f64; 10];
        for idx in 0..one_hot_vec.len() {
            one_hot_vec[idx] = (idx == class as usize) as i8 as f64;
        }
        one_hot_vec
    }
}

impl DataSet for MNISTDataSet {
    fn get_random_minibatch(&mut self, batch_size: usize) -> Batch {
        let mut batch_x_outer = vec![];
        let mut batch_y_outer = vec![];

        while batch_x_outer.len() < batch_size {
            let mut rand_idx = rand::random::<usize>() % self.batch_bank_x.data.len();
            batch_x_outer.push(self.batch_bank_x.data.remove(rand_idx));
            batch_y_outer.push(self.batch_bank_y.data.remove(rand_idx));
        }

        let batch_x_transpose = Matrix {
            rows: batch_size,
            cols: self.num_inputs,
            data: batch_x_outer
        };

        let batch_y_transpose = Matrix {
            rows: batch_size,
            cols: self.num_classes,
            data: batch_y_outer
        };

        Batch {
            size: batch_size,
            input: batch_x_transpose.transpose(),
            target: batch_y_transpose.transpose()
        }
    }

    fn replenish_minibatches(&mut self) {
        self.batch_bank_x = self.training_x.explicit_copy();
        self.batch_bank_y = self.training_y.explicit_copy();
    }

    fn get_training_set_size(&self) -> usize {
        self.training_set_size
    }

    fn get_validation_set(&self) -> Batch {
        Batch {
            size: self.validation_set_size,
            input: self.validation_x.explicit_copy(),
            target: self.validation_y.explicit_copy()
        }
    }

    fn get_test_set(&self) -> Batch {
        Batch {
            size: self.test_set_size,
            input: self.test_x.explicit_copy(),
            target: self.test_y.explicit_copy()
        }
    }

    fn get_num_inputs(&self) -> usize {
        self.num_inputs
    }
}

fn main() {
    let mut rdr = Reader::from_file("./data/train.csv").unwrap();
    let mut mnist_vec = vec![];
    let mut count = 0;

    println!("Creating MNIST Dataset...");
    for record in rdr.decode() {
        let record: MNISTRecord = record.unwrap();
        mnist_vec.push(record);
        count += 1;
        if count > 14_000 {
            break;
        }
    }

    let mnist = MNISTDataSet::new(mnist_vec, 10_000usize, 2_000usize, 2_000usize);
    println!("Done!");

    let mut model = Model::new(mnist);

    model.add(Layer::FullyConnected{ num_neurons: 784 });
    model.add(Layer::FullyConnected{ num_neurons: 100 });
    model.add(Layer::Softmax{ num_classes: 10 });

    let batch_size = 100;
    let num_epochs = 3;

    model.fit(batch_size, num_epochs);
}

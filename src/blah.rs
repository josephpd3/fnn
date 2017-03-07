#![allow(dead_code)]
extern crate ndarray;
extern crate csv;
extern crate rustc_serialize;
extern crate fnn;

//use csv::Reader;
use fnn::matrix::Matrix;

#[derive(RustcDecodable)]
struct MNISTRecord {
    data: Vec<u8>,
}

fn main() {
    // let mut rdr = Reader::from_file("./data/train.csv").unwrap();

    // for record in rdr.decode() {
    //     let record: MNISTRecord = record.unwrap();
    //     //println!("{:?}", record.data);
    // }

    let mut input = vec![];

    for x in 1..10 {
        input.push(x as f64);
    }

    let m = Matrix::new(2, 2, &vec![1f64, 2f64, 3f64, 4f64]);
    println!("{}", m);
    println!("{:?}", m.flatten());
    println!("{}", m.zero_pad(1));
    println!("{:?}", m[0]);
    println!("{:?}", m[0][0]);
    println!("{}", Matrix::zeroes(3, 3));
}

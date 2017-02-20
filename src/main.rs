extern crate ndarray;
extern crate csv;
extern crate rustc_serialize;
extern crate fnn;

use csv::Reader;

#[derive(RustcDecodable)]
struct MNISTRecord {
    data: Vec<u8>
}

fn main() {
    let mut rdr = Reader::from_file("./data/train.csv").unwrap();

    for record in rdr.decode() {
        let record: MNISTRecord = record.unwrap();
        println!("{:?}", record.data);
        break
    }
}
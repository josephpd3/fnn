/// Multiplies two vectors together if they are of
/// equal length
pub fn vector_mult(left: &Vec<f64>, right: &Vec<f64>) -> f64 {
    assert!(left.len() == right.len(), "Vectors not of equal length, can't multiply those!");
    let mut sum = 0f64;
    for idx in 0..left.len() {
        sum += left[idx] * right[idx];
    }
    sum
}
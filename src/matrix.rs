use rand;
use rand::distributions::{Normal, IndependentSample};

use std::fmt;
use std::ops::Index;
use std::ops::{Add, Sub, Mul};
use std::error::Error as StdError;

/// A macro for initializing Matrices
///
/// ```ignore
/// use fnn::matrix::Matrix;
///
/// let mat = matrix![1f64, 2f64, 3f64;
///                   4f64, 5f64, 6f64;
///                   7f64, 8f64, 9f64];
///
/// assert_eq!(mat, Matrix::new(3, 3, &vec![1f64, 2f64, 3f64,
///                                         4f64, 5f64, 6f64,
///                                         7f64, 8f64, 9f64]));
/// ```
#[macro_export]
macro_rules! matrix {
    ( $( $( $x:expr ),+ );+ ) => {{
        let mut num_rows = 0;
        let mut num_cols = 0;

        let mut outer = vec![];

        $(
            let mut inner = vec![];
            $(
                inner.push($x);
            )+
            outer.push(inner);
        )+

        Matrix {
            rows: outer.len(),
            cols: outer[0].len(),
            data: outer
        }
    }}
}

/// This struct represents Matrix comprised of data in a `Vec<Vec<f64>>` format.
/// This struct is designed to be used as an immutable facet in computation
/// to encourage a more functional programming style--which is idiomatic to Rust.
///
/// Design of this struct is largely focused on supplementing the development of
/// Convolutional Neural Networks for classification and recognition. This is largely
/// facilitated through functions such as [convolve_with](#method.convolve_with) and [zero_pad](#method.zero_pad).
///
#[derive(Debug)]
pub struct Matrix {
    /// rows in the matrix
    pub rows: usize,
    /// columns in the matrix
    pub cols: usize,
    /// data within the matrix
    pub data: Vec<Vec<f64>>
}

impl Matrix {

    /// Constructs a new `Matrix` comprised of floating-point 64-bit elements.
    ///
    /// # Arguments
    ///
    /// * `rows` - number of rows
    /// * `cols` - number of cols
    /// * `data` - data to be stored in the Matrix as a reference to a one-dimensional vector
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, &vec![1f64, 2f64, 3f64, 4f64]);
    ///
    /// let b = Matrix::new(3, 2, &vec![1f64, 2f64,
    ///                                 3f64, 4f64,
    ///                                 5f64, 6f64]);
    /// ```
    pub fn new(rows: usize, cols: usize, data: &Vec<f64>) -> Matrix {
        let mut outer = vec![];

        for i in 0..rows {
            let mut inner = vec![];

            for j in 0..cols {
                inner.push(*data.get((i * cols) + j).unwrap());
            }

            outer.push(inner);
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: outer
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Constructs a new `Matrix` comprised of floating-point
    /// 64-bit elements all initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `rows` - number of rows
    /// * `cols` - number of cols
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let zeroes = Matrix::zeroes(2, 2);
    ///
    /// assert_eq!(Matrix::zeroes(2, 2), Matrix::new(2, 2, &vec![0f64; 4]));
    /// ```
    pub fn zeroes(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows: rows,
            cols: cols,
            data: vec![vec![0f64; cols]; rows]
        }
    }

    /// Constructs a new `Matrix` of the given dimensions which
    /// has elements sampled from a Normal distribution between
    /// -1 and 1.
    ///
    /// # Arguments
    ///
    /// * `rows` - number of rows
    /// * `cols` - number of cols
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let zeroes = Matrix::zeroes(2, 2);
    ///
    /// assert_eq!(Matrix::zeroes(2, 2), Matrix::new(2, 2, &vec![0f64; 4]));
    /// ```
    pub fn gaussian(rows: usize, cols: usize) -> Matrix {
        let normal = Normal::new(0.0, 0.15);
        let mut outer = vec![];

        for i in 0..rows {
            let mut inner = vec![];

            for j in 0..cols {
                inner.push(normal.ind_sample(&mut rand::thread_rng()));
            }

            outer.push(inner);
        }

        Matrix {
            rows: rows,
            cols: cols,
            data: outer
        }
    }

    /// "Flattens" a Matrix into a `Vec<f64>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let v = vec![1f64, 2f64, 3f64, 4f64];
    ///
    /// let m = Matrix::new(2, 2, &v);
    ///
    /// assert_eq!(v, m.flatten());
    /// ```
    pub fn flatten(&self) -> Vec<f64> {
        let mut flat = vec![];
        for row in &self.data {
            flat.extend(row.iter().cloned());
        }
        flat
    }

    /// Transposes a Matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let v = vec![1f64, 2f64, 3f64, 4f64];
    ///
    /// let m = Matrix::new(2, 2, &v);
    ///
    /// assert_eq!(m.transpose(), Matrix::new(2, 2, &vec![1f64, 3f64,
    ///                                                   2f64, 4f64]));
    /// ```
    pub fn transpose(&self) -> Matrix {
        let mut outer = vec![];

        for col in 0..self.cols {
            let mut inner = vec![];

            for row in 0..self.rows {
                inner.push(self[row][col]);
            }

            outer.push(inner);
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: outer
        }
    }

    /// Apply a supplied function of the form `Fn(f64) -> f64` to
    /// every element of the Matrix, returning a new Matrix of the
    /// result
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let v = vec![1f64, 2f64, 3f64, 4f64];
    ///
    /// let m = Matrix::new(2, 2, &v);
    ///
    /// let after = m.mat_map(|x| x * 2f64);
    ///
    /// assert_eq!(after, Matrix::new(2, 2, &vec![2f64, 4f64,
    ///                                           6f64, 8f64]));
    /// ```
    pub fn mat_map<F>(&self, f: F) -> Matrix
        where F: Fn(f64) -> f64 
    {
        let mut outer = vec![];

        for row in 0..self.rows {
            let mut inner = vec![];

            for col in 0..self.cols {
                inner.push(f(self[row][col]));
            }

            outer.push(inner);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: outer
        }
    }

    /// Explicitly copies the Matrix. Be warned, this is likely not optimal!
    pub fn explicit_copy(&self) -> Matrix {
        let mut outer = vec![];

        //println!("Making an explicit copy of matrix with {} rows and {} cols", self.rows, self.cols);

        for row in 0..self.rows {
            let mut inner = vec![];

            for col in 0..self.cols {
                inner.push(self[row][col]);
            }

            outer.push(inner);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: outer
        }
    }

    /// Performs elementwise multiplication of Matrix with another Matrix
    ///
    pub fn ew_multiply(&self, other: &Matrix) -> Matrix {
        let mut outer = vec![];

        assert!(self.rows == other.rows, "Rows not equal in elementwise multiply!");
        assert!(self.cols == other.cols, "Rows not equal in elementwise multiply!");

        for row in 0..self.rows {
            let mut inner = vec![];

            for col in 0..self.cols {
                inner.push(self[row][col] * other[row][col]);
            }

            outer.push(inner);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: outer
        }
    }

    /// "Pads" a Matrix with a number of layers of zeroes equivalent to
    /// the designated padding. This function is primarily for use in
    /// Convolutional Neural Networks.
    ///
    /// # Arguments
    ///
    /// * `padding` - how much padding to add around the Matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// let v = vec![1f64, 2f64, 3f64, 4f64];
    ///
    /// let m = Matrix::new(2, 2, &v);
    ///
    /// assert_eq!(m.zero_pad(1), Matrix::new(4, 4, &vec![0f64, 0f64, 0f64, 0f64,
    ///                                                   0f64, 1f64, 2f64, 0f64,
    ///                                                   0f64, 3f64, 4f64, 0f64,
    ///                                                   0f64, 0f64, 0f64, 0f64]));
    /// ```
    
    pub fn zero_pad(&self, padding: usize) -> Matrix {
        let mut outer = vec![];
        let new_row_count = &self.rows + (2 * padding);
        let new_col_count = &self.cols + (2 * padding);

        for _ in 0..padding {
            outer.push(vec![0f64; new_col_count]);
        }

        for row in &self.data {
            let mut inner = vec![];

            inner.extend(vec![0f64; padding]);
            inner.extend(row.iter().cloned());
            inner.extend(vec![0f64; padding]);

            outer.push(inner);
        }

        for _ in 0..padding {
            outer.push(vec![0f64; new_col_count]);
        }
        
        Matrix {
            rows: new_row_count,
            cols: new_col_count,
            data: outer
        }
    }

    /// Given a filter Matrix, amount of zero-padding, stride, and 64-bit floating-point bias, perform convolution
    /// on self and return a new, output matrix. This function is primarily for use in Convolutional Neural Networks.
    ///
    /// # Arguments
    ///
    /// * `filter` - square Matrix to be used as convolution weights
    /// * `bias` - constant bias to apply to the convolution function
    /// * `padding` - amount of zero-padding to apply prior to convolution
    /// * `stride` - stride between filter convolutions
    ///
    /// # Examples
    ///
    /// ```
    /// use fnn::matrix::Matrix;
    ///
    /// /*
    /// Given a 4 x 4 matrix...
    /// [[1, 2, 0, 1]
    ///  [1, 0, 1, 2]
    ///  [0, 0, 2, 1],
    ///  [0, 2, 1, 1]]
    /// */
    ///
    /// let input = Matrix::new(4, 4, &vec![1f64, 2f64, 0f64, 1f64,
    ///                                     1f64, 0f64, 1f64, 2f64,
    ///                                     0f64, 0f64, 2f64, 1f64,
    ///                                     0f64, 2f64, 1f64, 1f64]);
    ///
    /// /*
    /// ...perform convolution with a 3 x 3 filter...
    /// [[0, 1, 0]
    ///  [1, 0, 1]
    ///  [0, 1, 0]]
    /// */
    ///
    /// let filter = Matrix::new(3, 3, &vec![0f64, 1f64, 0f64,
    ///                                      1f64, 0f64, 1f64,
    ///                                      0f64, 1f64, 0f64]);
    ///
    /// /*
    /// ...with bias 1, zero-padding 1, and stride of 1.
    /// */
    ///
    /// let bias = 1f64;
    /// let padding = 1usize;
    /// let stride = 1usize;
    /// let output = input.convolve_with(&filter, bias, padding, stride);
    ///
    /// /*
    /// The resultant output matrix should be the 4 x 4 matrix:
    /// [[4, 2, 5, 3]
    ///  [2, 5, 5, 4]
    ///  [2, 5, 4, 6]
    ///  [3, 2, 6, 3]]
    /// */
    ///
    /// assert_eq!(output, Matrix::new(4, 4, &vec![4f64, 2f64, 5f64, 3f64,
    ///                                            2f64, 5f64, 5f64, 4f64,
    ///                                            2f64, 5f64, 4f64, 6f64,
    ///                                            3f64, 2f64, 6f64, 3f64]));
    /// ```
    pub fn convolve_with(&self, filter: &Matrix, bias: f64, padding: usize, stride: usize) -> Matrix {
        let padded = &self.zero_pad(padding);
        let filter_size = filter.cols;
        let output_cols = ((&self.cols - filter_size + (2 * padding)) / stride) + 1;
        let output_rows = ((&self.rows - filter_size + (2 * padding)) / stride) + 1;

        let mut output_as_vec = vec![];

        // Calculate each component in the output matrix
        for i in 0..output_rows {
            for j in 0..output_cols {
                let mut sum = 0f64;

                // Convolve with the filter in position relative to input matrix
                for alpha in 0..filter_size {
                    for beta in 0..filter_size {
                        let translated_row = alpha + (stride * i);
                        let translated_col = beta + (stride * j);

                        sum += filter[alpha][beta] * padded[translated_row][translated_col]
                    }
                }

                output_as_vec.push(sum + bias);
            }
        }

        Matrix::new(output_rows, output_cols, &output_as_vec)
    }

    pub fn sum_cols(&self) -> Matrix {
        let mut outer = vec![];
        for row in 0..self.rows {
            outer.push(vec![self[row].iter().sum()]);
        }
        Matrix {
            rows: self.rows,
            cols: 1,
            data: outer
        }
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut inner = vec![];
        for col in 0..self.cols {
            let mut sum = 0f64;
            for row in 0..self.rows {
                sum += self[row][col]
            }
            inner.push(sum);
        }
        Matrix {
            rows: 1,
            cols: self.cols,
            data: vec![inner]
        }
    }

}

#[derive(Debug)]
pub enum Error {
    MatrixAddError,
    MatrixMultError
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::MatrixAddError => f.write_str("Error Adding Two Matrices"),
            Error::MatrixMultError => f.write_str("Error Multiplying Two Matrices")
        }
    }
}

impl StdError for Error {
    fn description(&self) -> &str {
        match *self {
            Error::MatrixAddError => "Error Adding Two Matrices",
            Error::MatrixMultError => "Error Multiplying Two Matrices"
        }
    }
}

/// Adds a Matrix to another Matrix
///
/// # Examples
///
/// ```
/// use fnn::matrix::Matrix;
///
/// let a = Matrix::new(2, 2, &vec![1f64, 2f64,
///                                 3f64, 4f64]);
///
/// let b = Matrix::new(2, 2, &vec![1f64, 2f64,
///                                 3f64, 4f64]);
///
/// let c = a + b;
///
/// assert_eq!(c, Matrix::new(2, 2, &vec![2f64, 4f64,
///                                       6f64, 8f64]));
/// ```
impl Add<Matrix> for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix {
        let mut output_as_vec = vec![];
        assert!(self.rows == other.rows, "Rows not equal in addition!");
        assert!(self.cols == other.cols, "Rows not equal in addition!");

        for row in 0..self.rows {
            for col in 0..self.cols {
                output_as_vec.push(self[row][col] + other[row][col]);
            }
        }

        Matrix::new(self.rows, self.cols, &output_as_vec)
    }
}

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, other: &'b Matrix) -> Matrix {
        let mut output_as_vec = vec![];
        assert!(self.rows == other.rows, "Rows not equal in addition!");
        assert!(self.cols == other.cols, "Rows not equal in addition!");

        for row in 0..self.rows {
            for col in 0..self.cols {
                output_as_vec.push(self[row][col] + other[row][col]);
            }
        }

        Matrix::new(self.rows, self.cols, &output_as_vec)
    }
}

/// SUBTRACT
impl Sub<Matrix> for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix {
        let mut output_as_vec = vec![];
        assert!(self.rows == other.rows, "Rows not equal in subtraction!");
        assert!(self.cols == other.cols, "Rows not equal in subtraction!");

        for row in 0..self.rows {
            for col in 0..self.cols {
                output_as_vec.push(self[row][col] - other[row][col]);
            }
        }

        Matrix::new(self.rows, self.cols, &output_as_vec)
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, other: &'b Matrix) -> Matrix {
        let mut output_as_vec = vec![];
        assert!(self.rows == other.rows, "Rows not equal in subtraction!");
        assert!(self.cols == other.cols, "Rows not equal in subtraction!");

        for row in 0..self.rows {
            for col in 0..self.cols {
                output_as_vec.push(self[row][col] - other[row][col]);
            }
        }

        Matrix::new(self.rows, self.cols, &output_as_vec)
    }
}

/// Multiplies a Matrix with another Matrix
///
/// # Examples
///
/// ```
/// use fnn::matrix::Matrix;
///
/// let a = Matrix::new(3, 2, &vec![1f64, 2f64,
///                                 3f64, 4f64,
///                                 5f64, 6f64]);
///
/// let b = Matrix::new(2, 2, &vec![1f64, 2f64,
///                                 3f64, 4f64]);
///
/// let c = a * b;
///
/// assert_eq!(c, Matrix::new(3, 2, &vec![7f64, 10f64,
///                                       15f64, 22f64,
///                                       23f64, 34f64]));
/// ```
impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert!(self.cols == other.rows, "LHS and RHS dims don't match for multiply!");
        let output_rows = self.rows;
        let output_cols = other.cols;
        let mut output_as_vec = vec![];

        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0f64;

                // Elementwise multiply self row by other column
                for i in 0..self.cols {
                    sum += self[row][i] * other[i][col];
                }

                output_as_vec.push(sum);
            }
        }

        Matrix::new(output_rows, output_cols, &output_as_vec)
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, other: &'b Matrix) -> Matrix {
        assert!(self.cols == other.rows, "LHS and RHS dims don't match for multiply!");
        let output_rows = self.rows;
        let output_cols = other.cols;
        let mut output_as_vec = vec![];

        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0f64;

                // Elementwise multiply self row by other column
                for i in 0..self.cols {
                    sum += self[row][i] * other[i][col];
                }

                output_as_vec.push(sum);
            }
        }

        Matrix::new(output_rows, output_cols, &output_as_vec)
    }
}

impl<'a, 'b> Mul<&'b Matrix> for &'a mut Matrix {
    type Output = Matrix;

    fn mul(self, other: &'b Matrix) -> Matrix {
        assert!(self.cols == other.rows, "LHS and RHS dims don't match for multiply!");
        let output_rows = self.rows;
        let output_cols = other.cols;
        let mut output_as_vec = vec![];

        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0f64;

                // Elementwise multiply self row by other column
                for i in 0..self.cols {
                    sum += self[row][i] * other[i][col];
                }

                output_as_vec.push(sum);
            }
        }

        Matrix::new(output_rows, output_cols, &output_as_vec)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        self.rows == other.rows
        && self.cols == other.cols
        && self.data == other.data
    }

    fn ne(&self, other: &Matrix) -> bool {
        self.rows != other.rows
        || self.cols != other.cols
        || self.data != other.data
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;

    fn index(&self, idx: usize) -> &Vec<f64> {
        &(&self.data[idx])
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut format_string = "[".to_string();
        let mut row_count = 1;

        for row in &self.data {
            if row_count != 1 {
                format_string = format_string + " ";
            }

            format_string = format_string + &format!("{:?}", row);

            if row_count != *&self.rows {
                format_string = format_string + "\n";
            }

            row_count += 1;
        }
        format_string = format_string + "]";

        write!(f, "{}", format_string)
    }
}

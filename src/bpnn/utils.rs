use super::types::Matrix;
use ndarray::Array;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

pub fn random_weight(row_size: usize, col_size: usize) -> Matrix {
    let range = Uniform::new(-1., 1.);
    let v: Vec<f64> = thread_rng()
        .sample_iter(&range)
        .take(row_size * col_size)
        .collect();

    Array::from_shape_vec((row_size, col_size), v).unwrap()
}

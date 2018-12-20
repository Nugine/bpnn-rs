use ndarray::{Array1, Array2};

pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

pub type Activation = fn(x: &Vector) -> Vector;
pub type DActivation = fn(y: &Vector) -> Vector;

pub type Cost = fn(target: &Vector, output: &Vector) -> f64;
pub type DCost = fn(target: &Vector, output: &Vector) -> Vector;

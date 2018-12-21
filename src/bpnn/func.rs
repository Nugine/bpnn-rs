use super::types::Vector;

pub fn tanh(x: &Vector) -> Vector {
    x.map(|f| f.tanh())
}

pub fn d_tanh(y: &Vector) -> Vector {
    y.map(|f| 1.0 - f * f)
}

pub fn sigmoid(x: &Vector) -> Vector {
    x.map(|f| 1.0 / (1.0 + (-f).exp()))
}

pub fn d_sigmoid(y: &Vector) -> Vector {
    y.map(|f| f * (1.0 - f))
}

pub fn relu(x: &Vector) -> Vector {
    x.map(|f| f.max(0.))
}

pub fn d_relu(y: &Vector) -> Vector {
    y.map(|f| if f.is_sign_positive() { 1. } else { 0. })
}

pub fn mse(target: &Vector, output: &Vector) -> f64 {
    target
        .iter()
        .zip(output.iter())
        .map(|(t, o)| (t - o) * (t - o))
        .sum::<f64>()
        / 2.0
}

pub fn d_mse(target: &Vector, output: &Vector) -> Vector {
    target - output
}

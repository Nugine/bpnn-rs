use super::types::Vector;

pub fn sigmoid(x: &Vector) -> Vector {
    x.map(|f| f.tanh())
}

pub fn d_sigmoid(y: &Vector) -> Vector {
    let mut r = y * y;
    r *= -1.0;
    r += 1.0;
    r
}

pub fn mse(target: &Vector, output: &Vector) -> f64 {
    target
        .iter()
        .zip(output.iter())
        .map(|(t, o)| {
            let x = t - o;
            x * x
        })
        .sum::<f64>()
        / 2.0
}

pub fn d_mse(target: &Vector, output: &Vector) -> Vector {
    target - output
}

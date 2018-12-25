mod bpnn;

fn main() {
    demo::run();
}

mod demo {
    use crate::bpnn::*;
    use ndarray::array;
    use std::iter::FromIterator;

    pub fn run() {
        let layer_settings: Vec<(usize, Activation, DActivation)> = vec![
            (3, tanh, d_tanh),
            (2, sigmoid, d_sigmoid),
            (1, relu, d_relu),
        ];

        let mut net = BPNN::new(2, &layer_settings, sse, d_sse);

        let patterns = vec![
            (array![0., 0.], array![0.]),
            (array![1., 0.], array![1.]),
            (array![0., 1.], array![1.]),
            (array![1., 1.], array![0.]),
        ];

        let rate = 0.5;
        let factor = 0.1;

        for i in 1..1001 {
            let total_error = net.train(&patterns, rate, factor);
            if i % 100 == 0 {
                println!("iteration: {:6}    error: {}", i, total_error);
            }
        }
        println!();

        let inputs = Vec::from_iter(patterns.into_iter().map(|(ip, _)| ip));
        let ops = net.predict(&inputs);
        for (ip, op) in inputs.into_iter().zip(ops.into_iter()) {
            println!("input: {}\noutput: {}\n", ip, op)
        }
    }
}

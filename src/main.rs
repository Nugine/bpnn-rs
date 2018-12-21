mod bpnn;

fn main() {
    demo::run();
}

mod demo {
    use crate::bpnn::*;
    use ndarray::array;

    pub fn run() {
        let layer_settings: Vec<(usize, Activation, DActivation)> = vec![
            (3, tanh, d_tanh),
            (2, sigmoid, d_sigmoid),
            (1, relu, d_relu),
        ];

        let mut net = BPNN::new(2, &layer_settings, mse, d_mse);

        let patterns = vec![
            (array![0., 0.], array![0.]),
            (array![1., 0.], array![1.]),
            (array![0., 1.], array![1.]),
            (array![1., 1.], array![0.]),
        ];

        let rate = 0.5;
        let factor = 0.1;

        for i in 1..1001 {
            let mut total_error = 0.;
            for (ip, tar) in &patterns {
                let (_, error) = net.train_once(ip, tar, rate, factor);
                total_error += error;
            }

            if i % 100 == 0 {
                println!("iteration: {:6}    error: {}", i, total_error);
            }
        }
        println!();

        for (ip, _) in &patterns {
            let op = net.predict(ip);
            println!("input: {}\noutput: {}\n", ip, op)
        }
    }
}

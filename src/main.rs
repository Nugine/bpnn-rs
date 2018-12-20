mod bpnn;

fn main() {
    println!("Hello, world!");
    demo::run();
}

mod demo {
    use crate::bpnn::*;
    use ndarray::array;

    pub fn run() {
        let layer_settings: Vec<(usize, Activation, DActivation)> =
            vec![(4, sigmoid, d_sigmoid), (1, sigmoid, d_sigmoid)];

        let mut net = BPNN::new(2, &layer_settings, mse, d_mse);

        let patterns = vec![
            (array![0., 0.], array![0.]),
            (array![1., 0.], array![1.]),
            (array![0., 1.], array![1.]),
            (array![1., 1.], array![0.]),
        ];

        let rate = 0.5;
        let factor = 0.1;

        for i in 0..1000 {
            let mut total_error = 0.;
            for (ip, tar) in &patterns {
                let (_, error) = net.train_once(ip, tar, rate, factor);
                // println!("input: {}\noutput: {}", ip, op);
                total_error += error;
            }

            if i % 100 == 0 && i >= 0 {
                println!("loss: {}\n", total_error);
            }
        }
    }
}

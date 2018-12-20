mod func;
mod types;
mod utils;

pub use self::func::*;
pub use self::types::*;
pub use self::utils::*;

use ndarray::Array;

pub struct BPNN {
    layer_num: usize,
    pub weights: Vec<Matrix>,
    pub changes: Vec<Matrix>,
    activations: Vec<Activation>,
    d_activations: Vec<DActivation>,
    cost: Cost,
    d_cost: DCost,
}

#[allow(non_snake_case)]
impl BPNN {
    pub fn new(
        input_size: usize,
        layer_settings: &Vec<(usize, Activation, DActivation)>,
        cost: Cost,
        d_cost: DCost,
    ) -> Self {
        let mut il = input_size + 1;
        let mut Ws: Vec<Matrix> = Vec::new();
        let mut Cs: Vec<Matrix> = Vec::new();
        let mut acts: Vec<Activation> = Vec::new();
        let mut d_acts: Vec<DActivation> = Vec::new();

        for (ol, act, d_act) in layer_settings {
            let ol = *ol;
            Ws.push(random_matrix(ol, il));
            Cs.push(zero_matrix(ol, il));
            acts.push(*act);
            d_acts.push(*d_act);
            il = ol;
        }

        Self {
            layer_num: layer_settings.len(),
            weights: Ws,
            changes: Cs,
            activations: acts,
            d_activations: d_acts,
            cost: cost,
            d_cost: d_cost,
        }
    }

    pub fn train_once(
        &mut self,
        input: &Vector,
        target: &Vector,
        rate: f64,
        factor: f64,
    ) -> (Vector, f64) {
        assert_eq!(input.len(), self.weights[0].dim().1 - 1);
        assert_eq!(target.len(), self.weights[self.layer_num - 1].dim().0);

        let l = self.layer_num;

        let W = &mut self.weights;

        let C = &mut self.changes;

        let activations = &self.activations;
        let d_activations = &self.d_activations;

        let mut a = vec![{
            let mut v = input.to_vec();
            v.push(1.);
            Array::from_vec(v)
        }];

        for i in 0..l {
            let x = &a[i];
            let y = W[i].dot(x);
            let act = (activations[i])(&y);
            a.push(act)
        }

        let mut d = vec![{
            let e = (self.d_cost)(target, &a[l]);
            let da = (d_activations[l - 1])(&a[l]);
            e * &da
        }];

        for i in 0..(l - 1) {
            let j = l - 1 - i;
            let e = W[j].t().dot(&d[i]);
            let da = (d_activations[j - 1])(&a[j]);
            d.push(e * &da)
        }

        d.reverse();

        for i in 0..l {
            W[i].scaled_add(factor, &C[i]);
        }

        let output = a.pop().unwrap();

        for i in (0..l).rev() {
            let (ol, il) = C[i].dim();
            let delta: Matrix = d.pop().unwrap().into_shape((ol, 1)).unwrap();
            let ip: Matrix = a.pop().unwrap().into_shape((1, il)).unwrap();
            C[i] = delta.dot(&ip);
        }

        for i in 0..l {
            W[i].scaled_add(rate, &C[i]);
        }

        let error = (self.cost)(target, &output);
        (output, error)
    }
}

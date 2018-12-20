mod types;
mod utils;

use self::types::*;
use self::utils::*;

use ndarray::Array;

pub struct BPNN {
    layer_num: usize,
    weights: Vec<Matrix>,
    changes: Vec<Matrix>,
    activations: Vec<Activation>,
    d_activations: Vec<DActivation>,
    cost: Cost,
    d_cost: DCost,
}

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
}

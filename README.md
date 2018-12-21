# bpnn-rs

 An implementation of **BPNN** in **Rust**
 
---

## Run it now

    git clone https://github.com/Nugine/bpnn-rs.git
    cd bpnn-rs
    cargo build
    cargo run

The neural network learned XOR function by training.

    iteration:    100    error: 0.353540086202427
    iteration:    200    error: 0.32357823287350707
    iteration:    300    error: 0.3097202209958916
    iteration:    400    error: 0.0023114803357387847
    iteration:    500    error: 0.00013376838445333176
    iteration:    600    error: 0.000005882603134813142
    iteration:    700    error: 0.00000021730736806563241
    iteration:    800    error: 0.000000007709918967308345
    iteration:    900    error: 0.0000000002713938800516082
    iteration:   1000    error: 0.000000000009539007440859954

    input: [0, 0]
    output: [0]

    input: [1, 0]
    output: [0.9999999518814118]

    input: [0, 1]
    output: [0.9999996256794317]

    input: [1, 1]
    output: [0.0000042431280589255715]

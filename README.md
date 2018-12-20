# bpnn-rs

 An implementation of **BPNN** in **Rust**
 
---

## Run it now

    git clone https://github.com/Nugine/bpnn-rs.git
    cd bpnn-rs
    cargo build
    cargo run

This neural network learned the XOR function by training.

    round: 0, loss: 1.0948453133681346
    round: 100, loss: 0.012072137341869835
    round: 200, loss: 0.002283495138420435
    round: 300, loss: 0.00473382896407484
    round: 400, loss: 0.0021332081154567654
    round: 500, loss: 0.0010898164807927476
    round: 600, loss: 0.0006329531738574197
    round: 700, loss: 0.0015553742817220394
    round: 800, loss: 0.0007531364151855329
    round: 900, loss: 0.0009064734253448057
    round: 1000, loss: 0.0010263287000098294

    input: [0, 0]
    output: [0.026425549461858377]

    input: [1, 0]
    output: [0.9909551058005059]

    input: [0, 1]
    output: [0.9908491211234489]

    input: [1, 1]
    output: [-0.03050681114398768]

use tch::nn;


pub fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}

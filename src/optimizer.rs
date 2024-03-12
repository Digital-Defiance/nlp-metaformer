use tch::{nn::{self, Optimizer, OptimizerConfig}, TchError};

use crate::config::Cli;



pub fn build_optimizer(vs:&nn::VarStore , config: &Cli) -> Optimizer {
    // https://paperswithcode.com/method/adam
    let adam: Result<Optimizer, TchError> = tch::nn::Adam::default().build(vs, config.learning_rate);
    match adam {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    }
}
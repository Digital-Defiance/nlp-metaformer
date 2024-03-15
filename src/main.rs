/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

*/

pub mod metaformer;
pub mod attention;
pub mod config;
pub mod mlflow;
pub mod files;
mod optimizer;
pub mod cuda;

use clap::Parser;
use optimizer::build_optimizer;
use nn::Optimizer;
use crate::mlflow::Metric;

use metaformer::{metaformer, MetaFormer};
use mlflow::log_metrics;

use tch;
use tch::Device;
use tch::nn;
use config::Cli;


impl Cli {
    pub fn get_device(&self) -> Device {
        let cuda = Device::cuda_if_available();
        if self.use_gpu == "True" {
            print!("Current training device: CUDA");
            match cuda {
                Device::Cuda(_) => cuda,
                _ => panic!("Invalid device specification. Did you mean CPU ?"),
            }
        } else if self.use_gpu == "False" {
            print!("Current training device: CPU");
            Device::Cpu
        } else {
            panic!("Invalid device configuration. Check USE_GPU env var.");
        }
    }
}



/// Implementation of gradient descent
fn main() {

    let config: Cli = Cli::parse();
    let training_device = config.get_device();
    let vs: nn::VarStore = nn::VarStore::new(training_device);

    let model: MetaFormer = metaformer(
        &vs.root(),
        config.dimension,
        config.input_vocabolary,
        config.context_window,
        config.get_device()
    ).build_model(&vs.root(), &config);

    let mut opt: Optimizer = build_optimizer(&vs, &config);

    for train_step in 1..(config.slices*config.epochs + 1) {
        let avg_train_loss = model.perform_train_step(&config, training_device, train_step, &mut opt);
        // let mut metrics: Vec<Metric> = model.perform_eval(&config, EVAL_SLICE_IDX, train_step);
        // metrics.push(avg_train_loss);
        log_metrics(&config, vec![avg_train_loss]);
    }


    let displacement = 5 + config.slices*config.epochs;
    for test_idx in 1..(config.slices + 1) {
        let step = test_idx + displacement;
        let metrics: Vec<Metric> = model.perform_eval(&config, test_idx, step);
        log_metrics(&config, metrics);
    }

}
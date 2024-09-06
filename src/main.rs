/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

*/

pub mod attention;
pub mod config;
pub mod cuda;
pub mod files;
pub mod metaformer;
pub mod mlflow;
mod optimizer;

use crate::mlflow::Metric;
use clap::Parser;
use libc::MPOL_DEFAULT;
use nn::Optimizer;
use optimizer::build_optimizer;

use metaformer::{metaformer, MetaFormer};
use mlflow::log_metrics;

use self::metaformer::embedder::create_embedder_module;
use clap::Parser;
use config::Cli;
use serde::Deserialize;
use tch;
use tch::nn;
use tch::Device;

#[derive(PartialEq, Clone, Copy, Deserialize, Debug)]
pub enum AttentionKind {
    Quadratic,
    ScaledDotProduct,
    Metric,
    Identity,
    AveragePooling,
}

/// Train a MetaFormer model.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[clap(long, env)]
    pub encoding: String,

    /// The kind of attention to use.
    #[clap(long, env)]
    pub attention_kind: String,

    /// Dimension of the vector space that the network uses internally to represent tokens
    #[clap(long, env)]
    pub dimension: i64,

    /// Number of transformer blocks
    #[clap(long, env)]
    pub depth: i64,

    /// Number of attention modules per transformer block
    #[clap(long, env)]
    pub heads: i64,

    /// Maximum number of tokens in the input sequence
    #[clap(long, env)]
    pub context_window: i64,

    /// Total number of tokens that the network recognizes in its input
    #[clap(long, env)]
    pub input_vocabolary: i64,

    /// Total number of tokens that the network recognizes in its outpput
    #[clap(long, env)]
    pub output_vocabolary: i64,

    /// Number of samples in a batch
    #[clap(long, env)]
    pub batch_size: i64,

    #[clap(long, env)]
    pub learning_rate: f64,

    #[clap(long, env)]
    pub epochs: i64,

    #[clap(long, env)]
    pub use_gpu: String,
}

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

    let model: MetaFormer = MetaFormer {
        embedding_dimension,
        training_device,
        size_of_context_window,
        layers: vec![Box::new(embedder)],
    }
    .build_model(vs_path.root(), &config, training_device);

    let mut opt: Optimizer = build_optimizer(&vs, &config);

    for train_step in 1..(config.slices * config.epochs + 1) {
        let avg_train_loss =
            model.perform_train_step(&config, training_device, train_step, &mut opt);
        // let mut metrics: Vec<Metric> = model.perform_eval(&config, EVAL_SLICE_IDX, train_step);
        // metrics.push(avg_train_loss);
        log_metrics(&config, vec![avg_train_loss]);
    }

    let displacement = 5 + config.slices * config.epochs;
    for test_idx in 1..(config.slices + 1) {
        let step = test_idx + displacement;
        let metrics: Vec<Metric> = model.perform_eval(&config, test_idx, step);
        log_metrics(&config, metrics);
    }
}

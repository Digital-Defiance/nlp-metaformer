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
use libc::printf;
use libc::MPOL_DEFAULT;
use nn::Optimizer;
use optimizer::build_optimizer;

use metaformer::{metaformer, MetaFormer};
use mlflow::log_metrics;

use clap::Parser;
use config::Cli;
use model::metaformer::embedder::create_embedder_module;
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
    print!("Reading CLI arguments");
    let config: Cli = Cli::parse();
    print!("Determining device, CPU or CUDA");
    let training_device = config.get_device();
    let vs: nn::VarStore = nn::VarStore::new(training_device);

    // Build model from config
    let model = {
        print!("Building model");
        let mut model: MetaFormer = MetaFormer {
            embedding_dimension,
            training_device,
            size_of_context_window,
            layers: vec![Box::new(embedder)],
        };
        println!("Creating embedder module");
        let embedder = create_embedder_module(
            vs_path,
            config.dimension,
            config.input_vocabolary,
            config.context_window,
            device,
        );

        for _ in 0..config.depth {
            print!("Adding transformer block...");
            model = match config.attention_kind.as_str() {
                quadratic => model.add_quadratic_form(vs_path, config.heads),
                scaled_dot_product => model.add_scaled_dot_product(vs_path, config.heads),
                identity => model,
                average_pooling => model.add_avg_pooling(vs_path, config.kernel_size.unwrap()),
                metric => todo!(),
                _ => panic!("Not suported"),
            };
            model = self.add_mlp(vs_path);
        }
        model.finish(vs_path, config.output_vocabolary);
        model
    };
    print!("Model has been built.");

    let adam: Result<Optimizer, TchError> =
        tch::nn::Adam::default().build(vs, config.learning_rate);
    let mut opt = match adam {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    };
    print!("Optimizer has been built");

    print!("Training will start now.");
    for epoch in 1..(config.epochs + 1) {
        print!("Performing training epoch");
        let mut loss_accumulator = MetricAccumulator::new("loss/train");

        println!("Reading data...");
        let data: std::collections::HashMap<String, tch::Tensor> = {
            let path_to_slice = std::path::Path::new(&config.path);
            let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
            dataslice.into_iter().collect()
        };
        let x_sc = data.get("X").unwrap().to(training_device);
        let y_s = data.get("Y").unwrap().to(training_device);

        println!("Loaded slice to device.");
        let s = y_s.size()[0]; // slice size s

        for idx in 0..(s / config.batch_size) {
            let start = idx * config.batch_size;
            let end = start + config.batch_size;

            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);

            let logits_bct = self.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false, kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);

            opt.backward_step(&loss);
            loss_accumulator.accumulate(loss.double_value(&[]));
        }

        let avg_train_loss = loss_accumulator.to_metric(train_step);
    }

    let displacement = 5 + config.slices * config.epochs;
    for test_idx in 1..(config.slices + 1) {
        let step = test_idx + displacement;
        let metrics: Vec<Metric> = model.perform_eval(&config, test_idx, step);
    }
}

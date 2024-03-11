


use std::str::FromStr;

use clap::Parser;
use tch::Device;
use serde::Deserialize;


#[derive(PartialEq, Clone, Copy, Deserialize)]
pub enum AttentionKind {
    Quadratic,
    ScaledDotProduct,
    Metric,
    Identity,
    AveragePooling,
}

/// Train a MetaFormer model.
#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {

    #[clap(long, env)]
    pub mlflow_tracking_username: String,

    #[clap(long, env)]
    pub mlflow_tracking_password: String,

    #[clap(long, env)]
    pub mlflow_run_id: String,

    #[clap(long, env)]
    pub mlflow_tracking_uri: String,

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
    pub slices: i64,

    #[clap(long, env)]
    pub epochs: i64,

    #[clap(long, env)]
    pub use_gpu: String,

    #[clap(long, env)]
    pub kernel_size: Option<i64>,

}


impl Cli {
    pub fn get_device(&self) -> Device {
        let cuda = Device::cuda_if_available();
        if self.use_gpu == "True" {
            print!("Current training device: CUDA");
            match cuda {
                Device::Cuda(_) => cuda,
                _ => todo!(),
            }
        } else if self.use_gpu == "False" {
            print!("Current training device: CPU");
            Device::Cpu
        } else {
            panic!("Invalid device configuration. Check USE_GPU env var.");
        }
    }
}

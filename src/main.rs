/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional


export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full


for static linking torch:
export LIBTORCH_STATIC=1
*/

use tch::kind;
use tch::nn::Module;
use tch::Device;
use tch::nn::OptimizerConfig;
use core::panic;
use std::thread;
use std::time::Duration;
use metaformer::MetaFormer;
use metaformer::AttentionKind;
pub mod metaformer;
pub mod attention;
use clap::Parser;
use tch::nn;
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::Deserialize;

/// Train a MetaFormer model.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // Path to config file
    #[arg(short, long)]
    path: String,
}



#[derive(Deserialize)]
struct Data {
    batch_size: i64,
    path: String
}


#[derive(Deserialize)]
struct Train {
    learning_rate: f64,
    data: Data,
}


#[derive(Deserialize)]
struct Model {
    /// The kind of attention to use.
    attention_kind: AttentionKind,
    /// Dimension of the vector space that the network uses internally to represent tokens 
    dimension: i64,
    /// Number of transformer blocks
    depth: i64,
    /// Number of attention modules per transformer block
    heads: i64,
    /// Maximum number of tokens in the input sequence
    context_window: i64,
    /// Total number of tokens that the network recognizes in its input
    input_vocabolary: i64,
    /// Total number of tokens that the network recognizes in its outpput
    output_vocabolary: i64,
}


#[derive(Deserialize)]
struct Config {
    use_gpu: bool,
    train: Train, 
    model: Model,
}



/// Implementation of gradient descent
/// https://paperswithcode.com/method/adam
fn main() {

    let config: Config = {
        let args: Cli = Cli::parse();
        let path = Path::new(&args.path);
        let content = std::fs::read_to_string(path).unwrap();
        let yaml = content.as_str();
        serde_yaml::from_str(yaml).unwrap()
    };

    let training_device = {
        let cuda = Device::cuda_if_available();
        if config.use_gpu {
            match cuda {
                Device::Cuda(_) => cuda,
                _ => todo!(),
            }
        } else {
            Device::Cpu
        }
    };

    let metaformer: MetaFormer = MetaFormer::new(
        config.model.dimension,
        config.model.depth,
        config.model.heads,
        config.model.context_window,
        config.model.input_vocabolary,
        config.model.output_vocabolary,
    );

    let vs: nn::VarStore = nn::VarStore::new(training_device);
    let vs_path: &nn::Path<'_> = &vs.root();
    let model = metaformer.create(vs_path, config.model.attention_kind);
    let mut opt: nn::Optimizer = tch::nn::Adam::default().build(&vs, config.train.learning_rate).unwrap();


    fn get_action() -> &'static str {
        let filename = "asdasddas";
        let path: &Path = Path::new(filename);
        while !path.exists() {
            thread::sleep(Duration::from_secs(5));
        }
        filename
    }

    const SLICE_PATH: &str = "slice.safetensors";
    
    loop {

        if get_action() == "stop" {
            break;
        }

        let dataslice: HashMap<String, tch::Tensor> = {
            let dataslice = tch::Tensor::read_safetensors(SLICE_PATH).unwrap();
            match fs::remove_file(SLICE_PATH) {
                Ok(_) => dataslice.into_iter().collect(),
                Err(e) => panic!("Error deleting file: {:?}", e),
            }
        };

        let x_sc = dataslice.get("X").unwrap().to(training_device);
        let y_s = dataslice.get("Y").unwrap().to(training_device);

        // let t = args.output_tokens;
        let s = y_s.size()[0]; // slice size s

        for idx in 0..(s / config.train.data.batch_size) {
            let start = idx*config.train.data.batch_size;
            let end = start + config.train.data.batch_size;

            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);

            let logits_bct = model.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);
            opt.backward_step(&loss);
        }
    }
}




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
use std::thread;
use std::time::Duration;
use metaformer::MetaFormer;
pub mod metaformer;
pub mod attention;
pub mod config;

use tch::nn;
use std::collections::HashMap;
use std::path::Path;
use std::fs;

use config::read_config;

const WAITING_TIMEOUT_SECONDS: u64 = 120; 
const WAIT_SECONDS: u64 = 5;

/// Implementation of gradient descent
fn main() {

    let config = read_config();

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

    // https://paperswithcode.com/method/adam
    let mut opt: nn::Optimizer = tch::nn::Adam::default().build(&vs, config.train.learning_rate).unwrap();
    let path_to_slice: &Path = Path::new(config.train.data.path_to_slice.as_str());

    loop {

        {
            println!("Waiting for file...");
            let mut wait = 0;
            while !path_to_slice.exists() {
                thread::sleep(Duration::from_secs(WAIT_SECONDS));
                wait += WAIT_SECONDS;
                if wait == WAITING_TIMEOUT_SECONDS {
                    eprintln!("Timed out while waiting for data.");
                    std::process::exit(1);
                }
            }
        }

        let dataslice: HashMap<String, tch::Tensor> = {
            let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
            match fs::remove_file(&config.train.data.path_to_slice) {
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




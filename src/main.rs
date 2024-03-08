/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional


export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full


for static linking torch:
export LIBTORCH_STATIC=1
*/

use std::str::FromStr;
use tch::kind;
use tch::nn::Module;
use tch::Device;
use tch::nn::OptimizerConfig;
use std::thread;
use std::time::Duration;
use metaformer::{AttentionKind, MetaFormer};
use mlflow::{MLFlowClient, MetricAccumulator };
pub mod metaformer;
pub mod attention;
pub mod config;
pub mod mlflow;

use tch;
use tch::nn;
use std::collections::HashMap;
use std::path::Path;
use std::fs;

// apt install libssh-dev -y


use config::{Cli, read_config};

fn wait(path: &Path) {
    let mut wait = 0;
    while !path.exists() {
        println!("File not found. Waiting {} seconds...", WAIT_SECONDS);
        thread::sleep(Duration::from_secs(WAIT_SECONDS));
        
        wait += WAIT_SECONDS;
        if wait == WAITING_TIMEOUT_SECONDS {
            eprintln!("Timed out while waiting for data.");
            std::process::exit(1);
        }
    };
}

const WAITING_TIMEOUT_SECONDS: u64 = 120; 
const WAIT_SECONDS: u64 = 5;

/// Implementation of gradient descent
fn main() {
    let config: Cli = read_config();
    let training_device = {
        let cuda = Device::cuda_if_available();
        if config.use_gpu {
            print!("Current training device: CUDA");
            match cuda {
                Device::Cuda(_) => cuda,
                _ => todo!(),
            }
        } else {
            print!("Current training device: CPU");
            Device::Cpu
        }
    };


    let metaformer: MetaFormer = MetaFormer::new(
        config.dimension,
        config.depth,
        config.heads,
        config.context_window,
        config.input_vocabolary,
        config.output_vocabolary,
    );

    let vs: nn::VarStore = nn::VarStore::new(training_device);
    let vs_path: &nn::Path<'_> = &vs.root();

    let attention_kind =  {
        if config.attention_kind == "Quadratic" {
            AttentionKind::Quadratic
        } else {
            AttentionKind::Quadratic
        }
    };

    let model = metaformer.create(vs_path, attention_kind, training_device);

    // https://paperswithcode.com/method/adam
    let mut opt: nn::Optimizer = tch::nn::Adam::default().build(&vs, config.learning_rate).unwrap();
    let mut global_idx: i64 = 0;


    fn read_slice(cli: &Cli, y_s: &mut tch::Tensor, x_sc: &mut tch::Tensor) {
        let path_to_slice: &Path = Path::new(&cli.path_to_slice);
        wait(path_to_slice);

        println!("Reading file...");

        let dataslice: HashMap<String, tch::Tensor> = {
            let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
            match fs::remove_file(path_to_slice) {
                Ok(_) => dataslice.into_iter().collect(),
                Err(e) => panic!("Error deleting file: {:?}", e),
            }
        };
        let x = dataslice.get("X").unwrap();
        let y = dataslice.get("Y").unwrap();

        let w = x.clone(x_sc);
        y.clone(y_s);

        println!("Slice has been loaded.");
    }

    loop {

        let avg_train_loss = {
            let mut x_sc = tch::Tensor::new();
            let mut y_s = tch::Tensor::new();

            // let y_s = opt_y_s.to(training_device);

            read_slice(&config, &mut y_s, &mut x_sc);


            println!("Loaded slice to device.");

            // let t = args.output_tokens;
            let s = y_s.size()[0]; // slice size s

            let mut loss_train = MetricAccumulator::new("loss/train");

            for idx in 0..(s / config.batch_size) {
                let start = idx*config.batch_size;
                let end = start + config.batch_size;

                let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
                let y_b = y_s.slice(0, start, end, 1);

                let logits_bct = model.forward(&x_bc);
                let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
                let loss = logits_bt.cross_entropy_for_logits(&y_b);
                
                opt.backward_step(&loss);
                loss_train.accumulate(loss.double_value(&[]));
            };

            loss_train.to_metric(global_idx)
        };

        let avg_eval_loss = {
            let mut loss_accumulator = MetricAccumulator::new("loss/eval");

            println!("Loading slice {}", global_idx);
            let path = format!("{}_{}", global_idx, config.path_to_slice);
            println!("{}", path);
            let path_to_slice: &Path = Path::new(path.as_str());


            {
                let mut wait = 0;
                while !path_to_slice.exists() {
                    println!("File not found. Waiting {} seconds...", WAIT_SECONDS);
                    thread::sleep(Duration::from_secs(WAIT_SECONDS));
                    
                    wait += WAIT_SECONDS;
                    if wait == WAITING_TIMEOUT_SECONDS {
                        eprintln!("Timed out while waiting for data.");
                        std::process::exit(1);
                    }
                }
            }
        
            println!("Reading file...");

            let dataslice: HashMap<String, tch::Tensor> = {
                let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
                match fs::remove_file(path_to_slice) {
                    Ok(_) => dataslice.into_iter().collect(),
                    Err(e) => panic!("Error deleting file: {:?}", e),
                }
            };


            println!("Slice has been loaded.");

            let x_sc = dataslice.get("X").unwrap().to(training_device);
            let y_s = dataslice.get("Y").unwrap().to(training_device);

            println!("Loaded slice to device.");

            // let t = args.output_tokens;
            let s = y_s.size()[0]; // slice size s

            for idx in 0..(s / config.batch_size) {
                let start = idx*config.batch_size;
                let end = start + config.batch_size;

                let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
                let y_b = y_s.slice(0, start, end, 1);

                let logits_bct = model.forward(&x_bc);
                let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
                let loss = logits_bt.cross_entropy_for_logits(&y_b);
                
                loss_accumulator.accumulate(loss.double_value(&[]));
            };

            loss_accumulator.to_metric(global_idx)
        };



        let run_id = String::from_str(config.mlflow_run_id.as_str()).unwrap();
        let url = format!("{}/api/2.0/mlflow/runs/log-batch", config.mlflow_db_uri);
        let mlflow_client = MLFlowClient { url, run_id };
        mlflow_client.log_metrics(vec![avg_eval_loss, avg_train_loss]);
        global_idx += 1;
    }
}



/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional


export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full


for static linking torch:
export LIBTORCH_STATIC=1
*/

use std::str::FromStr;
use anyhow::Result;
use tch::kind;
use tch::nn::Module;
use tch::Device;
use tch::nn::OptimizerConfig;
use std::thread;
use std::time::Duration;
use metaformer::{AttentionKind, MetaFormer};
pub mod metaformer;
pub mod attention;
pub mod config;

use tch;
use tch::nn;
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::{Deserialize, Serialize};

use reqwest;
// apt install libssh-dev -y


use config::{Cli, read_config};


#[derive(Serialize, Deserialize, Debug)]
struct Metric {
    key: String,
    value: f64,
    timestamp: i64,
    step: i64
}

impl Metric {
    fn new(key: String, value: f64, step: i64) -> Self {
        let timestamp = 11111;
        Self { key, value, timestamp, step }
    }
}


#[derive(Serialize, Deserialize, Debug)]
struct RequestBody {
    run_id: String,
    metrics: Vec<Metric>,
    step: i64,
}


fn send_request(run_id: String, metrics: Vec<Metric>, step: i64) {
    /*{
   "run_id": "2a14ed5c6a87499199e0106c3501eab8",
   "metrics": [
     {"key": "mae", "value": 2.5, "timestamp": 1552550804},
     {"key": "rmse", "value": 2.7, "timestamp": 1552550804},
   ],
   "params": [
     {"key": "model_class", "value": "LogisticRegression"},
   ]
}
 */
    let body = RequestBody {
        run_id: run_id,
        metrics: metrics,
        step,
    };

    let client = reqwest::blocking::Client::new();
     let resp = match client.post("http://127.0.0.1:5000/api/2.0/mlflow/runs/log-batch").json(&body).send() {
         Ok(resp) => resp.text().unwrap(),
         Err(err) => panic!("Error: {}", err)
     };
    println!("{}", resp)

}


const WAITING_TIMEOUT_SECONDS: u64 = 120; 
const WAIT_SECONDS: u64 = 5;

/// Implementation of gradient descent
fn main() {
    let config: Cli = read_config();
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

    loop {
        println!("Loading new slice...");

        let path = format!("{}_{}", global_idx, config.path_to_slice);
        println!("{}", path);
        let path_to_slice: &Path = Path::new(path.as_str());
        
        {

            let mut wait = 0;
            while !path_to_slice.exists() {
                thread::sleep(Duration::from_secs(WAIT_SECONDS));
                println!("Waiting for file...");
                wait += WAIT_SECONDS;
                if wait == WAITING_TIMEOUT_SECONDS {
                    eprintln!("Timed out while waiting for data.");
                    std::process::exit(1);
                }
            }
        }
    
        println!("Read file");

        let dataslice: HashMap<String, tch::Tensor> = {
            let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
            match fs::remove_file(path_to_slice) {
                Ok(_) => dataslice.into_iter().collect(),
                Err(e) => panic!("Error deleting file: {:?}", e),
            }
        };

        let x_sc = dataslice.get("X").unwrap().to(training_device);
        let y_s = dataslice.get("Y").unwrap().to(training_device);
        println!("Loaded slice to GPU.");

        // let t = args.output_tokens;
        let s = y_s.size()[0]; // slice size s

        let mut loss_cumul: f64 = 0.;

        for idx in 0..(s / config.batch_size) {
            let start = idx*config.batch_size;
            let end = start + config.batch_size;

            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);

            let logits_bct = model.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);
            
            opt.backward_step(&loss);

        
            let val = loss.double_value(&[]);
            loss_cumul += val;

        };
        
        let loss_train = Metric::new(
            String::from_str("loss/train").unwrap(),
            loss_cumul / ( (s / config.batch_size) as f64 ),
            global_idx
        );

        let run_id = String::from_str(config.mlflow_run_id.as_str()).unwrap();
        send_request(run_id, vec![loss_train], global_idx);
        global_idx += 1;
    }
}



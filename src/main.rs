/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

*/

pub mod metaformer;
pub mod attention;
pub mod config;
pub mod mlflow;
pub mod files;


use std::str::FromStr;
use clap::Parser;
use tch::{kind, nn::Module};
use tch::nn::OptimizerConfig;
use nn::Optimizer;
use tch::TchError;
use crate::mlflow::Metric;

use metaformer::{metaformer, MetaFormer};
use mlflow::{MLFlowClient, MetricAccumulator };

use tch;
use tch::nn;
use files::read_dataslice;
use config::Cli;


const QUADRATIC: &str = "quadratic";
const SCALED_DOT_PRODUCT: &str = "scaled_dot_product";
const IDENTITY: &str = "identity";
const AVERAGE_POOLING: &str = "average_pooling";
const METRIC: &str = "metric";



fn log_metrics(config: &Cli, metrics: Vec<Metric>) {
    let run_id: String = String::from_str(config.mlflow_run_id.as_str()).unwrap();
    let user: String = String::from_str(config.mlflow_tracking_username.as_str()).unwrap();
    let password: String = String::from_str(config.mlflow_tracking_password.as_str()).unwrap();

    let url = format!("{}/api/2.0/mlflow/runs/log-batch", config.mlflow_tracking_uri);
    let mlflow_client = MLFlowClient { url, run_id, user, password };
    mlflow_client.log_metrics(metrics);
}


fn build_model(vs_path: &nn::Path, mut model: MetaFormer, config: &Cli) -> MetaFormer {
    for _ in 0..config.depth {
        model = match config.attention_kind.as_str() {
            QUADRATIC => model.add_quadratic_form(vs_path, config.heads),
            SCALED_DOT_PRODUCT => model.add_scaled_dot_product(vs_path, config.heads),
            IDENTITY => model,
            AVERAGE_POOLING => model.add_avg_pooling(vs_path, config.kernel_size.unwrap()),
            METRIC => todo!(),
            _ => panic!("Not suported")
        };
        model = model.add_mlp(vs_path);
    }
    model.finish(vs_path, config.output_vocabolary)
}


fn build_optimizer(vs:&nn::VarStore , config: &Cli) -> Optimizer {
    // https://paperswithcode.com/method/adam
    let adam: Result<Optimizer, TchError> = tch::nn::Adam::default().build(vs, config.learning_rate);
    match adam {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    }
}

/// Implementation of gradient descent
fn main() {

    let config: Cli = Cli::parse();
    let training_device = config.get_device();
    let vs: nn::VarStore = nn::VarStore::new(training_device);
    let vs_path: &nn::Path = &vs.root();
    let model: MetaFormer = metaformer(
        vs_path,
        config.dimension,
        config.input_vocabolary,
        config.context_window,
        config.get_device()
    );
    let model: MetaFormer = build_model(vs_path, model, &config);
    let mut opt: Optimizer = build_optimizer(&vs, &config);


    for global_idx in 0..config.slices*config.epochs {


        let avg_train_loss = {
            let mut loss_accumulator = MetricAccumulator::new("loss/train");
        
            let dataslice: std::collections::HashMap<String, tch::Tensor> = read_dataslice(global_idx);
    
            let x_sc = dataslice.get("X").unwrap().to(training_device);
            let y_s = dataslice.get("Y").unwrap().to(training_device);
    
            println!("Loaded slice to device.");
    
            let s = y_s.size()[0]; // slice size s

            for idx in 0..(s / config.batch_size) {
                let start = idx*config.batch_size;
                let end = start + config.batch_size;

                let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
                let y_b = y_s.slice(0, start, end, 1);

                let logits_bct = model.forward(&x_bc);
                let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
                let loss = logits_bt.cross_entropy_for_logits(&y_b);
                
                opt.backward_step(&loss);
                loss_accumulator.accumulate(loss.double_value(&[]));
            };

            loss_accumulator.to_metric(global_idx)
        };

        let avg_eval_loss = {
            let mut loss_accumulator = MetricAccumulator::new("loss/eval");
            let mut acc_accumulator = MetricAccumulator::new("acc/eval");

            let dataslice: std::collections::HashMap<String, tch::Tensor> = read_dataslice(-1);

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

                let acc = logits_bt.accuracy_for_logits(&y_b);
                    
                loss_accumulator.accumulate(loss.double_value(&[]));
                acc_accumulator.accumulate(acc.double_value(&[]));

            };

            loss_accumulator.to_metric(global_idx)
        };

        log_metrics(&config, vec![avg_eval_loss, avg_train_loss]);
    }




    // TEST
    for test_idx in 2..config.slices {
        let mut loss_accumulator = MetricAccumulator::new("loss/test");
        let mut acc_accumulator = MetricAccumulator::new("acc/test");

        let metrics: Vec<Metric> = {
            
            let dataslice: std::collections::HashMap<String, tch::Tensor> = read_dataslice(-test_idx);
            let x_sc = dataslice.get("X").unwrap().to(training_device);
            let y_s = dataslice.get("Y").unwrap().to(training_device);
            println!("Loaded slice to device.");
            let s = y_s.size()[0]; // slice size s

            for idx in 0..(s / config.batch_size) {
                let start = idx*config.batch_size;
                let end = start + config.batch_size;

                let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
                let y_b: tch::Tensor = y_s.slice(0, start, end, 1);

                let logits_bct = model.forward(&x_bc);
                let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);


                let loss = logits_bt.cross_entropy_for_logits(&y_b);
                let acc = logits_bt.accuracy_for_logits(&y_b);
                
                
                loss_accumulator.accumulate(loss.double_value(&[]));
                acc_accumulator.accumulate(acc.double_value(&[]));

            };

            vec![
                loss_accumulator.to_metric(-1),
                acc_accumulator.to_metric(-1)
            ]
        };

        log_metrics(&config, metrics);

    }



}



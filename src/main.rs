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
use tch::{Device, TchError};
use crate::mlflow::Metric;

use metaformer::{metaformer, MetaFormer};
use mlflow::{MLFlowClient, MetricAccumulator };

use tch;
use tch::nn;
use files::read_dataslice;
use config::Cli;

const EVAL_SLICE_IDX: i64 = 0;

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




fn build_optimizer(vs:&nn::VarStore , config: &Cli) -> Optimizer {
    // https://paperswithcode.com/method/adam
    let adam: Result<Optimizer, TchError> = tch::nn::Adam::default().build(vs, config.learning_rate);
    match adam {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    }
}

impl MetaFormer {

    fn perform_eval(&self, config: &Cli, training_device: Device, slice_idx: i64, step: i64) -> Vec<Metric> {
        let _no_grad = tch::no_grad_guard();

        let mut loss_accumulator = MetricAccumulator::new("loss/test");
        let mut acc_accumulator = MetricAccumulator::new("acc/test");
    
        let dataslice: std::collections::HashMap<String, tch::Tensor> = read_dataslice(slice_idx);
    
        let x_sc = dataslice.get("X").unwrap().to(training_device);
        let y_s = dataslice.get("Y").unwrap().to(training_device);
    
        println!("Loaded slice to device.");
    
        // let t = args.output_tokens;
        let s = y_s.size()[0]; // slice size s
    
        for batch_idx in 0..(s / config.batch_size) {
            let start = batch_idx*config.batch_size;
            let end = start + config.batch_size;
    
            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);
    
            let logits_bct = self.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);
    
            let acc = logits_bt.accuracy_for_logits(&y_b);
                
            loss_accumulator.accumulate(loss.double_value(&[]));
            acc_accumulator.accumulate(acc.double_value(&[]));
    
        };

        vec![
            loss_accumulator.to_metric(step),
            acc_accumulator.to_metric(step)
        ]
    }

    fn build_model(mut self, vs_path: &nn::Path, config: &Cli) -> MetaFormer {
        for _ in 0..config.depth {
            self = match config.attention_kind.as_str() {
                QUADRATIC => self.add_quadratic_form(vs_path, config.heads),
                SCALED_DOT_PRODUCT => self.add_scaled_dot_product(vs_path, config.heads),
                IDENTITY => self,
                AVERAGE_POOLING => self.add_avg_pooling(vs_path, config.kernel_size.unwrap()),
                METRIC => todo!(),
                _ => panic!("Not suported")
            };
            self = self.add_mlp(vs_path);
        }
        self.finish(vs_path, config.output_vocabolary)
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

        let avg_train_loss = {
            let mut loss_accumulator = MetricAccumulator::new("loss/train");
            
            let dataslice: std::collections::HashMap<String, tch::Tensor> = read_dataslice(train_step);
    
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

            loss_accumulator.to_metric(train_step)
        };

        let mut metrics: Vec<Metric> = model.perform_eval(&config, training_device,  EVAL_SLICE_IDX, train_step);
        metrics.push(avg_train_loss);
        log_metrics(&config, metrics);
    }

    for test_idx in 1..config.slices {
        let metrics: Vec<Metric> = model.perform_eval(&config, training_device,  -test_idx, -test_idx);
        log_metrics(&config, metrics);
    }

}



/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

*/

pub mod metaformer;
pub mod attention;
pub mod config;
pub mod mlflow;
pub mod files;


use std::str::FromStr;
use tch::{kind, nn::Module};
use tch::nn::OptimizerConfig;

use metaformer::metaformer;
use mlflow::{MLFlowClient, MetricAccumulator };


use tch;
use tch::nn;
use files::read_dataslice;
use config::{Cli, read_config};


/// Implementation of gradient descent
fn main() {


    let config: Cli = read_config();


    let training_device = config.get_device();
    let vs: nn::VarStore = nn::VarStore::new(training_device);
    let vs_path: &nn::Path<'_> = &vs.root();

    let mut model = metaformer(
        vs_path,
        config.dimension,
        config.input_vocabolary,
        config.context_window,
        config.get_device()
    );

    let attention_kind = config.get_attention_kind();

    for _ in 0..config.depth {

        model = match attention_kind {
            config::AttentionKind::Quadratic => model.add_quadratic_form(vs_path, config.heads),
            config::AttentionKind::ScaledDotProduct => model.add_scaled_dot_product(vs_path, config.heads),
            config::AttentionKind::Metric => todo!(),
            config::AttentionKind::Identity => model,
            config::AttentionKind::AveragePooling => model.add_avg_pooling(vs_path, config.kernel_size.unwrap()),
        };

        model = model.add_mlp(vs_path);

    }

    model = model.finish(vs_path, config.output_vocabolary);


    let mut opt: nn::Optimizer = tch::nn::Adam::default().build(&vs, config.learning_rate).unwrap(); // https://paperswithcode.com/method/adam
    
    let total_slices: i64 = config.slices*config.epochs;
    for global_idx in 0..total_slices {
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

                // more validation stuff here, eventually
                
                loss_accumulator.accumulate(loss.double_value(&[]));
            };

            loss_accumulator.to_metric(global_idx)
        };

        let run_id: String = String::from_str(config.mlflow_run_id.as_str()).unwrap();
        let user: String = String::from_str(config.mlflow_tracking_username.as_str()).unwrap();
        let password: String = String::from_str(config.mlflow_tracking_password.as_str()).unwrap();

        let url = format!("{}/api/2.0/mlflow/runs/log-batch", config.mlflow_tracking_uri);
        let mlflow_client = MLFlowClient { url, run_id, user, password };
        mlflow_client.log_metrics(vec![avg_eval_loss, avg_train_loss]);
    }
}



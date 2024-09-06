use self::embedder::create_embedder_module;
use self::mlp::create_mlp;
use crate::attention::quadratic_form::QuadraticAttention;
use crate::attention::scaled_dot_product::ScaledDotProductAttention;
use crate::config::Cli;
use crate::files::read_dataslice;
use crate::mlflow::{Metric, MetricAccumulator};

pub mod commons;
pub mod embedder;
pub mod layer_norm;
pub mod mlp;

use crate::attention::avg_pooling::AvgPooling;
use commons::generate_init;
use layer_norm::create_layer_norm;
use tch;
use tch::nn::func;
use tch::nn::{Module, Optimizer};
use tch::Tensor;
use tch::{kind, nn, Device};

const QUADRATIC: &str = "quadratic";
const SCALED_DOT_PRODUCT: &str = "scaled_dot_product";
const IDENTITY: &str = "identity";
const AVERAGE_POOLING: &str = "average_pooling";
const METRIC: &str = "metric";

/// Defines structure of the metaformer model
/// GPT2 paper - https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
/// MetaFormer paper - TODO
#[derive(Debug)]
pub struct MetaFormer {
    pub embedding_dimension: i64,
    pub size_of_context_window: i64,
    pub training_device: Device,
    pub layers: Vec<Box<dyn Module>>,
}

impl Module for MetaFormer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.layers[0].forward(xs);
        self.layers
            .iter()
            .skip(1)
            .fold(xs, |xs, layer| layer.forward(&xs))
    }
}

impl MetaFormer {
    pub fn add_mlp(self, vs_path: &nn::Path) -> Self {
        let layer = create_mlp(vs_path, self.embedding_dimension);
        self.add(vs_path, layer)
    }

    pub fn add_avg_pooling(self, vs_path: &nn::Path, kernel_size: i64) -> Self {
        let layer = AvgPooling::new(kernel_size);
        self.add(vs_path, layer)
    }

    pub fn add_scaled_dot_product(self, vs_path: &nn::Path, number_of_heads: i64) -> Self {
        let layer = ScaledDotProductAttention::new(
            vs_path,
            number_of_heads,
            self.embedding_dimension,
            self.embedding_dimension / number_of_heads,
            self.size_of_context_window,
        );
        self.add(vs_path, layer)
    }

    pub fn add_quadratic_form(self, vs_path: &nn::Path, number_of_heads: i64) -> Self {
        let layer = QuadraticAttention::new(
            vs_path,
            number_of_heads,
            self.embedding_dimension,
            self.embedding_dimension / number_of_heads,
            self.size_of_context_window,
        );
        self.add(vs_path, layer)
    }

    pub fn finish(mut self, vs_path: &nn::Path, output_tokens: i64) -> Self {
        let d = self.embedding_dimension;
        let t = output_tokens;

        let linear_norm = create_layer_norm(vs_path, self.embedding_dimension);
        let projection_1dt = vs_path.var("projection_1dt", &[1, d, t], generate_init());

        let final_layer = nn::func(move |x_bcd| {
            let y_bcd = &linear_norm.forward(x_bcd);
            y_bcd.matmul(&projection_1dt)
        });

        self.layers.push(Box::new(final_layer));
        self
    }

    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    fn add<M: Module + 'static>(mut self, vs_path: &nn::Path, layer: M) -> Self {
        let layer_norm = create_layer_norm(vs_path, self.embedding_dimension);

        self.layers.push(Box::new(func(move |x: &tch::Tensor| {
            layer.forward(&layer_norm.forward(x)) + x
        })));
        self
    }
}

impl MetaFormer {
    pub fn perform_eval(&self, config: &Cli, slice_idx: i64, step: i64) -> Vec<Metric> {
        let _no_grad: tch::NoGradGuard = tch::no_grad_guard();

        let mut loss_accumulator = MetricAccumulator::new("loss/test");
        let mut acc_accumulator = MetricAccumulator::new("acc/test");

        let dataslice: std::collections::HashMap<String, tch::Tensor> =
            read_dataslice("test", slice_idx);

        let x_sc = dataslice.get("X").unwrap().to(self.training_device);
        let y_s = dataslice.get("Y").unwrap().to(self.training_device);

        println!("Loaded slice to device.");

        // let t = args.output_tokens;
        let s = y_s.size()[0]; // slice size s

        for batch_idx in 0..(s / config.batch_size) {
            let start = batch_idx * config.batch_size;
            let end = start + config.batch_size;

            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);

            let logits_bct = self.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false, kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);

            let acc = logits_bt.accuracy_for_logits(&y_b);

            loss_accumulator.accumulate(loss.double_value(&[]));
            acc_accumulator.accumulate(acc.double_value(&[]));
        }

        vec![
            loss_accumulator.to_metric(step),
            acc_accumulator.to_metric(step),
        ]
    }

    pub fn build_model(
        mut self,
        vs_path: &nn::Path,
        config: &Cli,
        device: tch::Device,
    ) -> MetaFormer {
        let embedder = create_embedder_module(
            vs_path,
            config.dimension,
            config.input_vocabolary,
            config.context_window,
            device,
        );

        for _ in 0..config.depth {
            self = match config.attention_kind.as_str() {
                QUADRATIC => self.add_quadratic_form(vs_path, config.heads),
                SCALED_DOT_PRODUCT => self.add_scaled_dot_product(vs_path, config.heads),
                IDENTITY => self,
                AVERAGE_POOLING => self.add_avg_pooling(vs_path, config.kernel_size.unwrap()),
                METRIC => todo!(),
                _ => panic!("Not suported"),
            };
            self = self.add_mlp(vs_path);
        }
        self.finish(vs_path, config.output_vocabolary)
    }

    pub fn perform_train_step(
        &self,
        config: &Cli,
        training_device: Device,
        train_step: i64,
        opt: &mut Optimizer,
    ) -> Metric {
        let mut loss_accumulator = MetricAccumulator::new("loss/train");
        let dataslice: std::collections::HashMap<String, tch::Tensor> =
            read_dataslice("train", train_step);
        let x_sc = dataslice.get("X").unwrap().to(training_device);
        let y_s = dataslice.get("Y").unwrap().to(training_device);
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

        loss_accumulator.to_metric(train_step)
    }
}

/// Creates a new empty metaformer layer.
pub fn metaformer(
    vs_path: &nn::Path,
    embedding_dimension: i64,
    size_of_vocabolary: i64,
    size_of_context_window: i64,
    training_device: tch::Device,
) -> MetaFormer {
    let embedder = create_embedder_module(
        vs_path,
        embedding_dimension,
        size_of_vocabolary,
        size_of_context_window,
        training_device,
    );

    MetaFormer {
        embedding_dimension,
        training_device,
        size_of_context_window,
        layers: vec![Box::new(embedder)],
    }
}

/*
pub struct MetaFormer {


    /// Dimension of the vector space that the network
    /// uses internally to represent tokens
    embedding_dimension: i64,

    /// Number of transformer blocks
    model_depth: i64,

    /// Number of attention modules per transformer block
    number_of_heads: i64,

    /// Maximum number of tokens in the input sequence
    size_of_context_window: i64,

    /// Total number of tokens that the network recognizes
    size_of_vocabolary: i64,

    output_tokens: i64,

    kernel_size: Option<i64>,


}

*/

/*
#[test]
pub fn test_model_creation(){

    let metaformer: MetaFormer = MetaFormer::new(
        32,
        3,
        2,
        10,
        5,
        5,
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let vs_path = &vs.root();
    let _quadratic_network = metaformer.create(vs_path, AttentionKind::Quadratic, Device::Cpu);
    // let _transformer_network = metaformer.create(vs_path, AttentionKind::ScaledDotProduct);
    // let _metric_network = metaformer.create(vs_path, AttentionKind::Metric);
}
 */

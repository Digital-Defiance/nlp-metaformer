/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

*/

use clap::Parser;
use nn::Optimizer;
use tch;
use tch::kind;
use tch::nn;
use tch::nn::Module;
use tch::nn::OptimizerConfig;
use tch::Device;
use tch::TchError;
use tch::Tensor;

/// Train a MetaFormer model.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[clap(long, env)]
    pub path: String,

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

#[derive(Debug)]
pub struct MetaFormer {
    pub layers: Vec<Box<dyn Module>>,
}

impl Module for MetaFormer {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let xs = self.layers[0].forward(xs);
        self.layers
            .iter()
            .skip(1)
            .fold(xs, |xs, layer| layer.forward(&xs))
    }
}

pub fn generate_init() -> nn::Init {
    tch::nn::init::DEFAULT_KAIMING_UNIFORM
}

pub fn create_layer_norm(vs_path: &nn::Path, embedding_dimension: i64) -> impl Module {
    let config = nn::LayerNormConfig {
        /*a value added to the denominator for numerical stability. Default: 1e-5 */
        eps: 1e-5,

        /*
        a boolean value that when set to True, this module has learnable
        per-element affine parameters initialized to ones (for weights)
        and zeros (for biases).
         */
        elementwise_affine: true,

        ws_init: generate_init(),
        bs_init: generate_init(),
        cudnn_enabled: false,
    };
    nn::layer_norm(vs_path, vec![embedding_dimension], config)
}

pub fn create_projection(vs: &nn::Path, d: i64, t: i64) -> impl nn::Module {
    let projection_1dt = vs.var("projection_1dt", &[1, d, t], generate_init());
    nn::func(move |y_bcd| y_bcd.matmul(&projection_1dt))
}

pub fn create_mlp(vs: &nn::Path, embedding_dimension: i64) -> impl nn::Module {
    let d: i64 = embedding_dimension;
    let q: i64 = embedding_dimension * 2;

    let projection_1dq = vs.var("projection_1dq", &[1, d, q], generate_init());
    let expansion_1qd = vs.var("expansion_1qd", &[1, q, d], generate_init());

    nn::func(move |input_bcd: &tch::Tensor| {
        let x_bcq = &input_bcd.matmul(&projection_1dq);
        let activations_bcq = x_bcq.gelu("none");
        activations_bcq.matmul(&expansion_1qd) // https://arxiv.org/abs/1512.03385
    })
}

/// Creates the embedder module, which transforms integer valued tokens into positionally encoded embeddings.
pub fn create_embedder_module(
    vs: &nn::Path,
    embedding_dimension: i64,
    size_of_vocabolary: i64,
    size_of_context_window: i64,
    training_device: tch::Device,
) -> impl nn::Module {
    let config = nn::EmbeddingConfig {
        /*
        If True, gradient w.r.t. weight matrix will be a sparse tensor.
        See Notes for more details regarding sparse gradients
         */
        sparse: false,

        // If given, this will scale gradients by the inverse of frequency of the words in the mini-batch
        scale_grad_by_freq: false,

        /*
        If specified, the entries at padding_idx do not contribute to the gradient;
        therefore, the embedding vector at padding_idx is not updated during training,
        i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding
        vector at padding_idx will default to all zeros, but can be updated to another
        value to be used as the padding vector.
         */
        padding_idx: 0,

        ws_init: generate_init(),
    };
    let d: i64 = embedding_dimension;
    let v: i64 = size_of_vocabolary;
    let c: i64 = size_of_context_window;

    let vocabolary_vd = nn::embedding(vs, v, d, config);
    let positional_encoding_cd = nn::embedding(vs, c, d, config);

    nn::func(move |x_bc: &tch::Tensor| {
        let indexes = tch::Tensor::arange(x_bc.size()[1], (tch::Kind::Int, training_device));
        vocabolary_vd.forward(&x_bc) + positional_encoding_cd.forward(&indexes)
    })
}

/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
/// https://arxiv.org/abs/2111.11418 - evidence that any of the attention mechanisms might have similar performance
// Implement the nn::Module trait for QuadraticAttention.
#[derive(Debug)]
pub struct QuadraticAttention {
    projections_1ndq: Tensor,
    metric_tensors_1nqq: Tensor,
    adapter_1pd: Tensor,
    sqrt_q: f64,
    cp: (i64, i64),
}
impl nn::Module for QuadraticAttention {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {
        let b = x_bcd.size()[0];
        // assert_eq!(x_bcd.size(), vec![b, c, d]);

        // Apply n projections to the input
        let x_b1cd = &x_bcd.unsqueeze(1);
        let x_bncq = &x_b1cd.matmul(&self.projections_1ndq);
        // debug_assert_eq!(x_bncq.size(), vec![b, n, c, q]);

        // Use n custom dot products to generate n score tables
        let x_bnqc = &x_bncq.transpose(-1, -2);
        let dotproducts_bncc = &x_bncq.matmul(&self.metric_tensors_1nqq.matmul(x_bnqc));
        // debug_assert!(dotproducts_bncc.size() == vec![b, n, c, c]);

        // From scaled dot product attention introduced in https://arxiv.org/abs/1706.03762
        let scaled_dotproducts_bncc = &dotproducts_bncc.divide_scalar(self.sqrt_q);

        let softmaxed_scaled_dotproducts_bncc =
            &scaled_dotproducts_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &x_bncq
            .transpose(-1, -2)
            .matmul(softmaxed_scaled_dotproducts_bncc);
        // debug_assert!(y_bnqc.size() == vec![b, n, q, c]);

        let y_bcp = &y_bnqc.reshape(&[b, self.cp.0, self.cp.1]);
        // debug_assert!(y_bcp.size() == vec![b, c, p]);

        y_bcp.matmul(&self.adapter_1pd)
    }
}

/// Performs self attention N times using the quadratic form $xW_nx.T$ where $W_n$ is a learnable matrix.
/// This is an early version of the metric self attention, where $W$ is forced to have the properties a metric tensor.
/// https://arxiv.org/abs/2111.11418 - evidence that any of the attention mechanisms might have similar performance
// Implement the nn::Module trait for QuadraticAttention.
#[derive(Debug)]
pub struct ScaledDotProductAttention {
    query_weights_1ndq: Tensor,
    key_weights_1ndq: Tensor,
    value_weights_1ndq: Tensor,
    adapter_1pd: Tensor,
    sqrt_q: f64,
    cp: (i64, i64),
}
impl nn::Module for ScaledDotProductAttention {
    fn forward(&self, x_bcd: &Tensor) -> Tensor {
        let b = x_bcd.size()[0];
        // assert_eq!(x_bcd.size(), vec![b, self.c, self.d]);

        // Apply n projections to the input
        let x_b1cd = &x_bcd.unsqueeze(1);

        let queries_bncq = &x_b1cd.matmul(&self.query_weights_1ndq);
        let keys_bncq = &x_b1cd.matmul(&self.key_weights_1ndq);
        let values_bncq = &x_b1cd.matmul(&self.value_weights_1ndq);

        // debug_assert_eq!(queries_bncq.size(), vec![b, n, c, q]);
        // debug_assert_eq!(keys_bncq.size(), vec![b, n, c, q]);
        // debug_assert_eq!(values_bncq.size(), vec![b, n, c, q]);

        // Use n custom dot products to generate n score tables
        let keys_bnqc = &keys_bncq.transpose(-1, -2);
        let scores_bncc = &queries_bncq.matmul(keys_bnqc);
        // debug_assert!(scores_bncc.size() == vec![b, n, c, c]);

        // From scaled dot product attention introduced in https://arxiv.org/abs/1706.03762
        let scaled_scores_bncc = &scores_bncc.divide_scalar(self.sqrt_q);

        let softmaxed_scaled_scores_bncc = &scaled_scores_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &values_bncq
            .transpose(-1, -2)
            .matmul(softmaxed_scaled_scores_bncc);
        // debug_assert!(y_bnqc.size() == vec![b, n, q, c]);

        let y_bcp = &y_bnqc.reshape(&[b, self.cp.0, self.cp.1]);
        // debug_assert!(y_bcp.size() == vec![b, c, p]);

        y_bcp.matmul(&self.adapter_1pd)
    }
}

fn main() {
    print!("Reading CLI arguments");
    let config: Cli = Cli::parse();
    print!("Determining device, CPU or CUDA");
    let cuda = Device::cuda_if_available();
    let training_device = match cuda {
        Device::Cuda(_) => cuda,
        _ => Device::Cpu,
    };

    let (x_sc, y_s) = {
        println!("Reading data...");
        let path_to_slice = std::path::Path::new(&config.path);
        let dataslice = tch::Tensor::read_safetensors(path_to_slice).unwrap();
        let data: std::collections::HashMap<String, tch::Tensor> = dataslice.into_iter().collect();
        let x_sc = data.get("X").unwrap().to(training_device);
        let y_s = data.get("Y").unwrap().to(training_device);
        (x_sc, y_s)
    };

    print!("Building model");
    let vs: nn::VarStore = nn::VarStore::new(training_device);
    let model = {
        let vs_path = &vs.root();

        println!("Creating embedder module");
        let embedder = create_embedder_module(
            &vs.root(),
            config.dimension,
            config.input_vocabolary,
            config.context_window,
            training_device,
        );
        let mut layers: Vec<Box<dyn Module>> = vec![Box::new(embedder)];

        let n = config.heads;
        let d = config.dimension;
        let c = config.context_window;
        let q = config.dimension / config.heads;
        let p = config.dimension;
        for _ in 0..config.depth {
            print!("Adding transformer block...");
            print!("Adding linear norm");
            let linear_norm = create_layer_norm(&vs.root(), config.dimension);
            layers.push(Box::new(linear_norm));

            print!("Adding attention module");
            let layer: Box<dyn nn::Module> = match config.attention_kind.as_str() {
                "quadratic" => Box::new(QuadraticAttention {
                    projections_1ndq: vs_path.var(
                        "projections_1ndq",
                        &[1, n, d, q],
                        generate_init(),
                    ),
                    metric_tensors_1nqq: vs_path.var(
                        "metric_tensors_1nqq",
                        &[1, n, q, q],
                        generate_init(),
                    ),
                    adapter_1pd: vs_path.var("adapter_1pd", &[1, p, d], generate_init()),
                    sqrt_q: f64::sqrt(q as f64),
                    cp: (c, p),
                }),
                "scaled_dot_product" => Box::new(ScaledDotProductAttention {
                    query_weights_1ndq: vs_path.var(
                        "query_weights_1ndq",
                        &[1, n, d, q],
                        generate_init(),
                    ),
                    key_weights_1ndq: vs_path.var(
                        "key_weights_1ndq",
                        &[1, n, d, q],
                        generate_init(),
                    ),
                    value_weights_1ndq: vs_path.var(
                        "value_weights_1ndq",
                        &[1, n, d, q],
                        generate_init(),
                    ),
                    adapter_1pd: vs_path.var("adapter_1pd", &[1, p, d], generate_init()),
                    sqrt_q: f64::sqrt(q as f64),
                    cp: (c, p),
                }),
                // identity => model,
                // average_pooling => layers.push_avg_pooling(&vs.root(), config.kernel_size.unwrap()),
                // metric => todo!(),
                _ => panic!("Not suported"),
            };
            layers.push(layer);

            println!("Adding MLP");
            let mlp = create_mlp(&vs.root(), config.dimension);
            layers.push(Box::new(mlp));
        }

        print!("Adding linear norm");
        let linear_norm = create_layer_norm(&vs.root(), config.dimension);
        layers.push(Box::new(linear_norm));

        print!("adding final layer");
        let final_layer = create_projection(&vs.root(), config.dimension, config.output_vocabolary);
        layers.push(Box::new(final_layer));

        MetaFormer { layers }
    };
    print!("Model has been built.");

    let mut opt = match tch::nn::Adam::default().build(&vs, config.learning_rate) {
        Ok(result) => result,
        Err(err) => panic!("Error while building optimizer: {}", err),
    };
    print!("Optimizer has been built");

    print!("Training will start now.");
    let s = y_s.size()[0]; // slice size s
    for _epoch in 1..(config.epochs + 1) {
        print!("Performing training epoch");
        // let mut loss_accumulator = MetricAccumulator::new("loss/train");

        for idx in 0..(s / config.batch_size) {
            let start = idx * config.batch_size;
            let end = start + config.batch_size;

            let x_bc: tch::Tensor = x_sc.slice(0, start, end, 1);
            let y_b = y_s.slice(0, start, end, 1);

            let logits_bct = model.forward(&x_bc);
            let logits_bt = logits_bct.mean_dim(1, false, kind::Kind::Float);
            let loss = logits_bt.cross_entropy_for_logits(&y_b);

            opt.backward_step(&loss);
            // loss_accumulator.accumulate(loss.double_value(&[]));
        }

        // let avg_train_loss = loss_accumulator.to_metric(train_step);
    }

    // let displacement = 5 + config.slices * config.epochs;
    // for test_idx in 1..(config.slices + 1) {
    //     let step = test_idx + displacement;
    //     // let metrics: Vec<Metric> = model.perform_eval(&config, test_idx, step);
    // }
}

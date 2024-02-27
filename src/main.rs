/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

    TODO: output tokenizer


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

use metaformer::MetaFormer;
use metaformer::AttentionKind;
pub mod metaformer;
pub mod attention;
use clap::Parser;
use tch::nn;

/// Train a MetaFormer model.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The kind of attention to use.
    #[arg(short, long)]
    attention_kind: String,

    #[arg(short, long)]
    path: String,

    /// Dimension of the vector space that the network
    /// uses internally to represent tokens 
    #[arg(short, long)]
    embedding_dimension: i64,

    /// Number of transformer blocks
    #[arg(short, long)]
    model_depth: i64,

    /// Number of attention modules per transformer block
    #[arg(short, long)]
    number_of_heads: i64,

    /// Maximum number of tokens in the input sequence
    #[arg(short, long)]
    size_of_context_window: i64,

    /// Total number of tokens that the network recognizes
    #[arg(long)]
    size_of_vocabolary: i64,

    #[arg(short, long)]
    output_tokens: i64,
}



/*

cargo run 
  --attention-kind "quadratic"
  --path "."
  --embedding-dimension 10
  --model-depth 1
  --number-of-heads 2
  --size-of-context-window 400
  --size-of-vocabolary 60000
  --output-tokens 2


*/

fn find_tensor_by_name(key: String, slice: & Vec<(String, tch::Tensor)>) -> &tch::Tensor {
    let mut tmp: Option<&tch::Tensor> = None;
    for (name, tensor) in slice {
        if *name == key {
            tmp = Some(tensor);
            break;
        }
    }

    match tmp {
        Some(tensor) => tensor, 
        None => panic!("No value found"),
    }
}



/// Implementation of gradient descent
/// https://paperswithcode.com/method/adam
fn main() {

    let vs = nn::VarStore::new(Device::Cpu);
    let vs_path = &vs.root();

    let args: Cli = Cli::parse();
    let metaformer: MetaFormer = MetaFormer::new(
        args.embedding_dimension,
        args.model_depth,
        args.number_of_heads,
        args.size_of_context_window,
        args.size_of_vocabolary,
        args.output_tokens,
    );

    let kind = if args.attention_kind == "quadratic" {
        AttentionKind::Quadratic
    } else {
        AttentionKind::Quadratic
    };

    let model = metaformer.create(vs_path, kind);
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-3).unwrap();

    for epoch in 0..args.epochs {
        for slice in 0..args.slices {
            let filename = format!("epoch_{}_slice_{}.safetensors", epoch, slice);
            let slice: Vec<(String, tch::Tensor)> = tch::Tensor::read_safetensors("epoch_0_slice_0.safetensors").unwrap();
            let x_sc = find_tensor_by_name(String::from('X'), &slice);
            let y_s = find_tensor_by_name(String::from('Y'), &slice);
            
            // slice size 
            let s = y_s.size()[0];
            let t = args.output_tokens;
            let b = 32;

            for start in 0..s..b {
                let end = start + b;
                let x_bc = x_sc.slice(0, 0, b, 1);
                let y_b = y_s.slice(0, 0, b, 1);
            
                let logits_bct = model.forward(&x_bc);
                let logits_bt = logits_bct.mean_dim(1, false,  kind::Kind::Float);
                let loss = logits_bt.cross_entropy_for_logits(&y_b);
                opt.backward_step(&loss);
            }
        }

    }
}



















#[cfg(test)]
mod tests {
    use super::*; 
    use tch::{nn, Device};

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
        let _quadratic_network = metaformer.create(vs_path, AttentionKind::Quadratic);
        let _transformer_network = metaformer.create(vs_path, AttentionKind::ScaledDotProduct);
        let _metric_network = metaformer.create(vs_path, AttentionKind::Metric);
    }

}

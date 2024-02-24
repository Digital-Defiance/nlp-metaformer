/*

https://paperswithcode.com/paper/bert-pre-training-of-deep-bidirectional

    TODO: output tokenizer


export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
export LIBTORCH_USE_PYTORCH=1
export RUST_BACKTRACE=full
*/

use tch::Device;


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
    let optimizer = nn::adamw(
        0.9,
        0.9,
        0.1,
    );

    for epoch in 1..10 {
        for slice in 1..10 {
            // let data = load_slice(slice);
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

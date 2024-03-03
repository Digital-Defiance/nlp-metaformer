


use serde::Deserialize;
use clap::Parser;
use std::path::Path;
use colored::*;



use crate::metaformer::AttentionKind;

/// Train a MetaFormer model.
#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    // Path to config file
    #[arg(short, long)]
    pub path: String,
}



#[derive(Deserialize)]
pub struct Data {
    pub batch_size: i64,
    pub path_to_slice: String
}


#[derive(Deserialize)]
pub struct Train {
    pub learning_rate: f64,
    pub data: Data,
}


#[derive(Deserialize)]
pub struct Model {
    /// The kind of attention to use.
    pub attention_kind: AttentionKind,

    /// Dimension of the vector space that the network uses internally to represent tokens 
    pub dimension: i64,

    /// Number of transformer blocks
    pub depth: i64,

    /// Number of attention modules per transformer block
    pub heads: i64,

    /// Maximum number of tokens in the input sequence
    pub context_window: i64,

    /// Total number of tokens that the network recognizes in its input
    pub input_vocabolary: i64,

    /// Total number of tokens that the network recognizes in its outpput
    pub output_vocabolary: i64,
}


#[derive(Deserialize)]
pub struct Config {
    pub use_gpu: bool,
    pub train: Train, 
    pub model: Model,
}

pub fn read_config() -> Config {
    let args: Cli = Cli::parse();
    let path = Path::new(&args.path);

    if !path.exists() {
        eprintln!("File does not exist: {}", args.path);
        std::process::exit(1);
    };

    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("{} - Failed to read the file -> {}", "Error".red(), e);
            std::process::exit(1);
        }
    };

    match serde_yaml::from_str(&content) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("{} - Failed to parse the configuration -> {}",  "Error".red(),  e);
            std::process::exit(1);
        }
    }
    
}

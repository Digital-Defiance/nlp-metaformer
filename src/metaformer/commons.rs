
/// Defines structure of the quadratic attention model
/// GPT2 paper - https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
pub struct MetaformerParameters {

    /// Dimension of the vector space that the network
    /// uses internally to represent tokens 
    pub embedding_dimenson: i64,

    /// Number of transformer blocks
    pub model_depth: i64,

    /// Number of attention modules per transformer block
    pub number_of_heads: i64,

    /// Maximum number of tokens in the input sequence
    pub size_of_context_window: i64,

    /// Total number of tokens that the network recognizes
    pub size_of_vocabolary: i64,
}

pub fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 1. }
}


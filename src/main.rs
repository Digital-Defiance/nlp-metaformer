use tch::{nn::{self, Module}, Device};



fn embedding_table(vs_path: &nn::Path, num_embeddings: i64, embedding_dim: i64) -> impl Module {
    let ws_init = nn::Init::Randn { mean: 0., stdev: 3. };
    let config = nn::EmbeddingConfig{
        sparse: false,
        scale_grad_by_freq: false,
        ws_init: ws_init,
        padding_idx: 1
    };
    nn::embedding(vs_path, num_embeddings, embedding_dim, config)
}

fn embedder_module(vs_path: &nn::Path, embedding_dimenson: i64, size_of_context_window: i64) -> impl Module {

    let vocabolary = embedding_table(
        vs_path,
        60_000, 
        embedding_dimenson
    );

    let positional_encoding = embedding_table(
        vs_path, 
        size_of_context_window, 
        embedding_dimenson
    );

    nn::func(move |xs| vocabolary.forward(xs) + positional_encoding.forward(xs))
}


fn attention_module(){

}

fn linear_feedforward() {

}


fn transformer_module(){

}


struct MetricTensorNetworkParameters {
    embedding_dimenson: i64,
    size_of_context_window: i64,
    size_of_vocabolary: i64
}



fn metric_tensor_network(vs_path: &nn::Path, hyper_parameters: &MetricTensorNetworkParameters) -> impl Module {
    nn::seq().add(
        embedder_module(
            vs_path,
            hyper_parameters.embedding_dimenson,
            hyper_parameters.size_of_context_window,
        )
    )
}


fn main() {
    let vs: nn::VarStore = nn::VarStore::new(Device::Cpu);
    let vs_path = &vs.root();

    let network_config = MetricTensorNetworkParameters{
        embedding_dimenson: 256,
        size_of_context_window: 300,
        size_of_vocabolary: 60_000,
    };
    let _net = metric_tensor_network(vs_path, &network_config);
}
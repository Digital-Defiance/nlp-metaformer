
use tch;
use tch::nn::{self, Module };


// export LD_LIBRARY_PATH=/workspace/.pyenv_mirror/user/current/lib/python3.12/site-packages/torch/lib
// export LIBTORCH_USE_PYTORCH=1
// export RUST_BACKTRACE=full



struct ModelParameters {
    embedding_dimenson: i64,
    size_of_context_window: i64,
    size_of_vocabolary: i64,
    number_of_heads: i64
}

fn generate_init() -> nn::Init {
    nn::Init::Randn { mean: 0., stdev: 3. }
}

fn embedding_table(vs_path: &nn::Path, num_embeddings: i64, embedding_dim: i64) -> impl nn::Module {
    let config = nn::EmbeddingConfig{
        sparse: false,
        scale_grad_by_freq: false,
        ws_init: generate_init(),
        padding_idx: 0
    };
    nn::embedding(vs_path, num_embeddings, embedding_dim, config)
}

fn embedder_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {

    let vocabolary = embedding_table(
        vs_path,
        hyper_parameters.size_of_vocabolary, 
        hyper_parameters.embedding_dimenson
    );

    let positional_encoding = embedding_table(
        vs_path, 
        hyper_parameters.size_of_context_window,
        hyper_parameters.embedding_dimenson
    );

    nn::func(move |x_bc| {
        vocabolary.forward(&x_bc) + positional_encoding.forward(&x_bc)
    })
}


fn self_attention_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {

    let n = hyper_parameters.number_of_heads;
    let c = hyper_parameters.size_of_context_window;
    let d = hyper_parameters.embedding_dimenson;
    let q = hyper_parameters.embedding_dimenson /  hyper_parameters.number_of_heads;

    let projections_1ndq = vs_path.var("projections_1ndq", &[1, n, d, q], generate_init());
    let metric_tensors_1nqq = vs_path.var("metric_tensors_1nqq", &[1, n, q, q], generate_init());
    let mixer_1dd = vs_path.var("mixer_1dd", &[1, d, d], generate_init());

    // let sqrt_q: f32 = unsafe { sqrtf32(q) };

    nn::func(move |x_bcd| {
        // Apply n projections to the input 
        let x_b1cd = &x_bcd.unsqueeze(1);
        let x_bncq = &x_b1cd.matmul(&projections_1ndq);

        // Use n custom dot products to generate n score tables
        let x_bnqc = &x_bncq.transpose(-1, -2);
        let x_bncc = &x_bncq.matmul(&metric_tensors_1nqq.matmul(x_bnqc));
    
        // x_bnqq = &x_bnqq.divide_scalar(sqrt_q);

        let softmaxed_x_bncc = &x_bncc.softmax(-1, tch::kind::Kind::Float);
        let y_bnqc = &x_bncq.transpose(-1, -2).matmul(softmaxed_x_bncc);
        let y_bcd = &y_bnqc.reshape(x_bcd.size());
    
        y_bcd.matmul(&mixer_1dd)
    })
}




fn transformer_module(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {
    nn::seq()
    .add(self_attention_module(vs_path, hyper_parameters))
}


fn metric_tensor_network(vs_path: &nn::Path, hyper_parameters: &ModelParameters) -> impl nn::Module {
    let embedding_layer = embedder_module(vs_path, hyper_parameters);
    nn::seq().add(embedding_layer).add(transformer_module(vs_path, hyper_parameters))
}


fn main() {

    let vs: nn::VarStore = nn::VarStore::new(tch::Device::Cpu);
    let vs_path = &vs.root();

    let network_config = &ModelParameters {
        embedding_dimenson: 6,
        size_of_context_window: 30,
        size_of_vocabolary: 10,
        number_of_heads: 2,
    };

    let _net = metric_tensor_network(vs_path, network_config);

    let test_self_attention_module = self_attention_module(vs_path, network_config);


    let input = tch::Tensor::rand(
        &[9, network_config.size_of_context_window, network_config.embedding_dimenson],
        (tch::Kind::Float, tch::Device::Cpu)
    );

    let output = test_self_attention_module.forward(&input);
    for x in output.size(){
        print!("{}, ", x);
    }


    


}
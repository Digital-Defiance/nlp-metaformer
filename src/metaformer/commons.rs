use tch::nn;


pub fn generate_init() -> nn::Init {
    tch::nn::init::DEFAULT_KAIMING_UNIFORM
}

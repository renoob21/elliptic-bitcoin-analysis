use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    optim::AdamConfig,
};
use concat_aggregate::{model::ModelConfig, training::TrainingConfig};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "./tmp/output";
    concat_aggregate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );
}

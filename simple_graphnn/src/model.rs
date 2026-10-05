use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};
use elliptic_dataset::data::GraphBatch;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    node_linear: Linear<B>,
    neighbor_linear: Linear<B>,
    activation: Relu,
    output_layer: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    #[config(default = 2)]
    num_classes: usize,
    #[config(default = 256)]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            node_linear: LinearConfig::new(165, self.hidden_size).init(device),
            neighbor_linear: LinearConfig::new(165, self.hidden_size).init(device),
            activation: Relu::new(),
            output_layer: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, batch: GraphBatch<B>) -> Tensor<B, 2> {
        let node_x = self.node_linear.forward(batch.node_features);
        let neighbor_x = self.neighbor_linear.forward(batch.neighbor_features);

        let x = node_x + neighbor_x;
        let x = self.activation.forward(x);

        self.output_layer.forward(x)
    }
}

use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::*,
};
use elliptic_dataset::data::GraphBatch;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
    dropout: Dropout,
    output_layer: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    #[config(default = 2)]
    num_classes: usize,
    #[config(default = 512)]
    hidden_size: usize,
    #[config(default = "0.4")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(330, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size / 2).init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            output_layer: LinearConfig::new(self.hidden_size / 2, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, batch: GraphBatch<B>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![batch.node_features, batch.neighbor_features], 1);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }
}

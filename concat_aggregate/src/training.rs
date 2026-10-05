use std::fs;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{ClassReduction, FBetaScoreMetric, PrecisionMetric, RecallMetric},
    },
};
use elliptic_dataset::data::{GraphBatch, GraphData};

use crate::model::{Model, ModelConfig};

impl<B: Backend> Model<B> {
    pub fn forward_classification(&self, batch: GraphBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(batch.clone());

        let weights = vec![10.0, 1.0];

        let loss = CrossEntropyLossConfig::new()
            .with_weights(Some(weights))
            .init(&output.device())
            .forward(output.clone(), batch.targets.clone());
        ClassificationOutput {
            loss,
            output,
            targets: batch.targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = GraphBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_classification(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = GraphBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    fs::remove_dir_all(artifact_dir).ok();
    fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config.save(format!("{artifact_dir}/config.json")).unwrap();

    B::seed(&device, config.seed);

    let batcher = GraphData::load(34);
    let train = InMemDataset::new(batcher.train());
    let test = InMemDataset::new(batcher.test());

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((
            PrecisionMetric::multiclass(1, ClassReduction::Macro),
            RecallMetric::multiclass(1, ClassReduction::Macro),
            FBetaScoreMetric::multiclass(1.0, 1, ClassReduction::Macro),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .unwrap();
}

use std::fs;

use anyhow::Result;
use bigram::{BigramBatcher, BigramModelConfig};
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataloader::DataLoaderBuilder,
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{BasicOps, Tensor, TensorData, TensorKind, backend::AutodiffBackend},
    train::{
        ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use dataset::{TrainingDataset, unique_chars};

mod bigram;
mod dataset;

const DATASET: &str = include_str!("tiny-shakespeare.txt");
const OUTPUT_DIR: &str = "output";

type NGBackend = Wgpu;
type NGAutodiffBackend = Autodiff<NGBackend>;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: BigramModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 4)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: B::Device,
    train_data: &str,
    valid_data: &str,
    vocab: &Vec<char>,
) -> Result<()> {
    B::seed(1337);

    let trainer = BigramBatcher::<B> {
        device: device.clone(),
    };

    let validator = BigramBatcher::<B::InnerBackend> {
        device: device.clone(),
    };

    let trainer_loader = DataLoaderBuilder::new(trainer)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TrainingDataset::new(train_data, vocab));

    let validator_loader = DataLoaderBuilder::new(validator)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TrainingDataset::new(valid_data, vocab));

    let learner = LearnerBuilder::<
        B,
        ClassificationOutput<B>,
        ClassificationOutput<B::InnerBackend>,
        _,
        _,
        _,
    >::new(OUTPUT_DIR)
    .metric_train_numeric(AccuracyMetric::new())
    .metric_valid_numeric(AccuracyMetric::new())
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(CompactRecorder::new())
    .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 1 },
    ))
    .devices(vec![device.clone()])
    .num_epochs(config.num_epochs)
    .summary()
    .build(
        config.model.init(&device),
        config.optimizer.init(),
        config.learning_rate,
    );

    let model = learner.fit(trainer_loader, validator_loader);
    model.save_file(format!("{}/model", OUTPUT_DIR), &CompactRecorder::new())?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let training_len = DATASET.len() * 9 / 10;
    let training_data = &DATASET[0..training_len];
    let valid_data = &DATASET[training_len + 1..];
    let vocab = unique_chars(DATASET);

    let device = WgpuDevice::DefaultDevice;
    let config = TrainingConfig::new(BigramModelConfig::new(), AdamWConfig::new());

    fs::create_dir_all(OUTPUT_DIR)?;

    config.save(format!("{}/config.json", OUTPUT_DIR))?;

    train::<NGAutodiffBackend>(config, device, training_data, valid_data, &vocab)?;

    Ok(())
}

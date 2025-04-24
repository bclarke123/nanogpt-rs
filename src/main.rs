use std::fs;

use anyhow::Result;
use bigram::{BigramBatcher, BigramModel, BigramModelConfig, BigramModelRecord};
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataloader::DataLoaderBuilder,
    optim::AdamWConfig,
    prelude::*,
    record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{activation::softmax, backend::AutodiffBackend},
    train::{
        ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use dataset::{TrainingDataset, decode, sample_distribution, unique_chars};

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

    let trainer = BigramBatcher {
        // device: device.clone(),
    };

    let validator = BigramBatcher {
        // device: device.clone(),
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
    model.save_file(
        format!("{}/model", OUTPUT_DIR),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
    )?;

    Ok(())
}

fn generate<B: Backend>(device: B::Device, vocab: &Vec<char>, max_new_token: usize) -> Result<()> {
    let config = TrainingConfig::load(format!("{}/config.json", OUTPUT_DIR))?;

    let record: BigramModelRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{}/model", OUTPUT_DIR).into(), &device)?;

    let model: BigramModel<B> = config.model.init(&device).load_record(record);

    let start = vec![rand::random_range(0..vocab.len()) as i32];
    let mut input = Tensor::<B, 1, Int>::from_data(TensorData::from(start.as_slice()), &device);

    for _ in 0..max_new_token {
        let [input_dim] = input.dims();
        let logits = model.forward(input.clone().reshape([1, input_dim]));
        let [b, t, c] = logits.dims();
        let probs: Tensor<B, 2> = softmax(logits.slice([0..b, t - 1..t, 0..c]).squeeze(1), 1);
        let data = probs.to_data();
        let prob_elems = data.as_slice::<f32>().unwrap();
        let elem_next = sample_distribution(prob_elems);
        let input_next =
            Tensor::<B, 1, Int>::from(TensorData::from([elem_next as i32])).to_device(&device);
        input = Tensor::cat(vec![input, input_next], 0);
    }

    let data = input.to_data();
    let output = data.as_slice().unwrap();

    println!("{}", decode(output, vocab));

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

    // train::<NGAutodiffBackend>(config, device, training_data, valid_data, &vocab)?;

    generate::<NGAutodiffBackend>(device, &vocab, 500)?;

    Ok(())
}

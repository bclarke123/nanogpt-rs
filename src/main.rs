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

fn get_batch<B, const D: usize, K>(
    data: &[u32],
    block_size: usize,
    batch_size: usize,
) -> (Tensor<B, D, K>, Tensor<B, D, K>)
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
{
    let idx = rand::random_range(0..data.len() - block_size);

    let x = Tensor::stack::<D>(
        (0..batch_size)
            .map(|_| Tensor::<B, 1, K>::from(TensorData::from(&data[idx..idx + block_size])))
            .collect::<Vec<_>>(),
        0,
    );

    let y = Tensor::stack::<D>(
        (0..batch_size)
            .map(|_| {
                Tensor::<B, 1, K>::from(TensorData::from(&data[idx + 1..idx + block_size + 1]))
            })
            .collect::<Vec<_>>(),
        0,
    );

    (x, y)
}

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

    // let chars = unique_chars(&dataset);

    // println!("Unique characters: {}", String::from_iter(chars.iter()));

    // let encoded = encode("hii there", &chars);
    // println!("Encoded: {:?}", encoded);

    // let decoded = decode(&encoded, &chars);
    // println!("Decoded: {}", decoded);

    // let encoded = encode(&dataset, &chars);
    // let len = encoded.len() * 9 / 10;

    // let train_data = &encoded[..len];
    // let val_data = &encoded[len + 1..];

    // println!(
    //     "Using {len} bytes to train, {} to validate",
    //     encoded.len() - len
    // );

    // let batch_size = 4;
    // let block_size = 8;

    // let (xb, yb) = get_batch::<Wgpu, 2, Int>(train_data, block_size, batch_size);

    // println!("inputs: {:?} / {:?}", xb.shape(), xb.to_string());
    // println!("targets: {:?} / {:?}", yb.shape(), yb.to_string());

    // println!("{:?}", &encoded[..=block_size]);

    // let device = Default::default();
    // let tensor_data = TensorData::new(encoded[..len].to_vec(), [len; 1]);

    // let tensor: Tensor<Backend, 1, Int> =
    //     Tensor::<Backend, 1, Int>::from_data(tensor_data, &device);

    Ok(())
}

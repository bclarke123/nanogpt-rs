use crate::bigram::{BigramBatcher, BigramModel, BigramModelConfig, BigramModelRecord};
use crate::dataset::{TrainingDataset, decode, encode, sample_distribution};
use anyhow::Result;
use burn::{
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

pub fn train<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    train_data: &str,
    valid_data: &str,
    vocab: &Vec<char>,
) -> Result<()> {
    B::seed(config.seed);

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
    >::new(output_dir)
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
        format!("{}/model", output_dir),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
    )?;

    Ok(())
}

pub fn generate<B: Backend>(
    output_dir: &str,
    device: B::Device,
    vocab: &Vec<char>,
    max_new_token: usize,
) -> Result<()> {
    let config = TrainingConfig::load(format!("{}/config.json", output_dir))?;

    let record: BigramModelRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{}/model", output_dir).into(), &device)?;

    let model: BigramModel<B> = config.model.init(&device).load_record(record);

    let start = encode(" ", vocab);
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

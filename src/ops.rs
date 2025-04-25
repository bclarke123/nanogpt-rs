use crate::bigram::{BigramBatcher, BigramModel, BigramModelConfig, BigramModelRecord};
use crate::dataset::{TrainingDataset, decode, multinomial_distrib};
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
use rand::rng;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: BigramModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 3.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    train_data: &str,
    valid_data: &str,
    vocab: &[char],
) -> Result<()> {
    B::seed(config.seed);

    let trainer = BigramBatcher {};
    let validator = BigramBatcher {};

    let trainer_loader = DataLoaderBuilder::new(trainer)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TrainingDataset::new(
            train_data,
            config.model.block_size,
            vocab,
        ));

    let validator_loader = DataLoaderBuilder::new(validator)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TrainingDataset::new(
            valid_data,
            config.model.block_size,
            vocab,
        ));

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
    vocab: &[char],
    max_new_token: usize,
) -> Result<()> {
    let mut rand = rng();
    let config = TrainingConfig::load(format!("{}/config.json", output_dir))?;

    let record: BigramModelRecord<B> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(format!("{}/model", output_dir).into(), &device)?;

    let model: BigramModel<B> = config.model.init(&device).load_record(record);

    let start = vec![0];
    let mut input = Tensor::<B, 1, Int>::from_data(TensorData::from(start.as_slice()), &device);

    let block_size = config.model.block_size;

    for _ in 0..max_new_token {
        let [input_dim] = input.dims();

        let start_index = if input_dim > block_size {
            input_dim - block_size
        } else {
            0
        };

        let context_input = input.clone().slice([start_index..input_dim]);
        let [context_len] = context_input.dims();
        let model_input = context_input.reshape([1, context_len]);

        let logits = model.forward(model_input);
        let [b, t, c] = logits.dims();
        let probs: Tensor<B, 2> = softmax(logits.slice([0..b, t - 1..t, 0..c]).squeeze(1), 1);
        let data = probs.to_data();
        let prob_elems = data
            .as_slice::<f32>()
            .expect("Softmax did not return f32 data");

        let elem_next = multinomial_distrib(prob_elems, &mut rand);
        let input_next =
            Tensor::<B, 1, Int>::from(TensorData::from([elem_next as i32])).to_device(&device);
        input = Tensor::cat(vec![input.clone(), input_next], 0);
    }

    let data = input.to_data();
    let output = data.as_slice().unwrap();

    println!("{}", decode(output, vocab));

    Ok(())
}

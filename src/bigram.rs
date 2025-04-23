use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig},
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::TrainingItem;

#[derive(Config)]
pub struct BigramModelConfig {
    #[config(default = 65)]
    vocab_size: usize,

    #[config(default = 130)]
    d_model: usize,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        BigramModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct BigramModel<B: Backend> {
    embedding: Embedding<B>,
    linear: Linear<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input);
        self.linear.forward(x)
    }

    pub fn forward_classification(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(input);
        let [b, t, c] = output.dims();
        let [b_prim, t_prim] = targets.dims();
        let output = output.reshape([b * t, c]);
        let targets = targets.reshape([b_prim * t_prim]);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BigramBatcher {}

#[derive(Clone, Debug)]
pub struct BigramBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<TrainingItem, BigramBatch<B>> for BigramBatcher {
    fn batch(&self, items: Vec<TrainingItem>) -> BigramBatch<B> {
        let inputs = items
            .iter()
            .map(|i| Tensor::<B, 1, Int>::from(TensorData::from(i.context.as_slice())))
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|i| Tensor::<B, 1, Int>::from(TensorData::from(i.target.as_slice())))
            .collect::<Vec<_>>();

        let inputs = Tensor::stack(inputs, 0);
        let targets = Tensor::stack(targets, 0);

        BigramBatch { inputs, targets }
    }
}

impl<B: AutodiffBackend> TrainStep<BigramBatch<B>, ClassificationOutput<B>> for BigramModel<B> {
    fn step(&self, item: BigramBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item.inputs, item.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BigramBatch<B>, ClassificationOutput<B>> for BigramModel<B> {
    fn step(&self, item: BigramBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item.inputs, item.targets)
    }
}

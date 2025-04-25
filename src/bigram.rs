use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::{
            MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
        },
        loss::CrossEntropyLossConfig,
    },
    prelude::Backend,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Clone, Debug)]
pub struct TrainingItem {
    pub context: Vec<i32>,
    pub target: Vec<i32>,
}

#[derive(Config)]
pub struct BigramModelConfig {
    #[config(default = 8)]
    block_size: usize,

    #[config(default = 65)]
    vocab_size: usize,

    #[config(default = 32)]
    d_model: usize,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        BigramModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.d_model).init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            sa_head: MultiHeadAttentionConfig::new(self.d_model, 4).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct BigramModel<B: Backend> {
    embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    linear: Linear<B>,
    sa_head: MultiHeadAttention<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = input.dims();
        let device = input.device();

        let tok_emb = self.embedding.forward(input);

        let pos_indices = Tensor::<B, 1, Int>::arange(0..t as i64, &device);
        let pos_indices = pos_indices.reshape([1, t]);

        let pos_emb = self.position_embedding.forward(pos_indices);

        let x = tok_emb + pos_emb;

        let mask = generate_autoregressive_mask(b, t, &device);
        let mha_input = MhaInput::self_attn(x).mask_attn(mask);
        let attn_output = self.sa_head.forward(mha_input);
        let x = attn_output.context;

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
pub struct BigramBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
pub struct BigramBatcher {}

impl BigramBatcher {
    fn stack<'a, B: Backend, T: Fn(&'a TrainingItem) -> &'a [i32]>(
        items: &'a [TrainingItem],
        op: T,
    ) -> Tensor<B, 2, Int> {
        let tensors = items
            .iter()
            .map(op)
            .map(|arr| Tensor::<B, 1, Int>::from(arr))
            .collect::<Vec<_>>();
        Tensor::stack(tensors, 0)
    }
}

impl<B: Backend> Batcher<TrainingItem, BigramBatch<B>> for BigramBatcher {
    fn batch(&self, items: Vec<TrainingItem>) -> BigramBatch<B> {
        let inputs = Self::stack::<B, _>(&items, |ti| ti.context.as_slice());
        let targets = Self::stack::<B, _>(&items, |ti| ti.target.as_slice());

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

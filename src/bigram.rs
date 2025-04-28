use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
        attention::{
            MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
        },
        loss::CrossEntropyLossConfig,
    },
    prelude::Backend,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct BlockConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            sa_head: MultiHeadAttentionConfig::new(self.d_model, self.n_heads).init(device),

            // Feed forwards
            ffwd_linear1: LinearConfig::new(self.d_model, self.d_model * 4).init(device),
            ffwd_linear2: LinearConfig::new(self.d_model * 4, self.d_model).init(device),
            ffwd_activation: Gelu::new(),
            ffwd_dropout: DropoutConfig::new(self.dropout).init(),

            ln1: LayerNormConfig::new(self.d_model).init(device),
            ln2: LayerNormConfig::new(self.d_model).init(device),
            attn_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    sa_head: MultiHeadAttention<B>,
    ffwd_linear1: Linear<B>,
    ffwd_linear2: Linear<B>,
    ffwd_activation: Gelu,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
    attn_dropout: Dropout,
    ffwd_dropout: Dropout,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _c] = x.dims();
        let device = x.device();

        let x_norm1 = self.ln1.forward(x.clone());
        let mask = generate_autoregressive_mask(b, t, &device);
        let mha_input = MhaInput::self_attn(x_norm1).mask_attn(mask);
        let mha_output = self.sa_head.forward(mha_input);
        let mha_drop = self.attn_dropout.forward(mha_output.context);
        let x = x + mha_drop;

        let x_norm2 = self.ln2.forward(x.clone());
        let x_ffwd1 = self.ffwd_linear1.forward(x_norm2);
        let ffwd_act = self.ffwd_activation.forward(x_ffwd1);
        let ffwd_output = self.ffwd_linear2.forward(ffwd_act);
        let ffwd_drop = self.ffwd_dropout.forward(ffwd_output);

        x + ffwd_drop
    }
}

#[derive(Clone, Debug)]
pub struct TrainingItem {
    pub context: Vec<i32>,
    pub target: Vec<i32>,
}

#[derive(Config)]
pub struct BigramModelConfig {
    #[config(default = 128)]
    pub block_size: usize,

    #[config(default = 65)]
    vocab_size: usize,

    #[config(default = 192)]
    d_model: usize,

    #[config(default = 4)]
    n_heads: usize,

    #[config(default = 4)]
    n_layers: usize,

    #[config(default = 0.2)]
    dropout: f64,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        let bc = BlockConfig::new(self.d_model, self.n_heads).with_dropout(self.dropout);
        BigramModel {
            embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.d_model).init(device),
            embedding_dropout: DropoutConfig::new(self.dropout).init(),
            blocks: (0..self.n_layers).map(|_| bc.init(device)).collect(),
            final_ln: LayerNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct BigramModel<B: Backend> {
    embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    embedding_dropout: Dropout,
    blocks: Vec<Block<B>>,
    final_ln: LayerNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_b, t] = input.dims();
        let device = input.device();

        let tok_emb = self.embedding.forward(input);

        let pos_indices = Tensor::<B, 1, Int>::arange(0..t as i64, &device);
        let pos_indices = pos_indices.reshape([1, t]);
        let pos_emb = self.position_embedding.forward(pos_indices);

        let mut x = tok_emb + pos_emb;

        x = self.embedding_dropout.forward(x);

        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        x = self.final_ln.forward(x);

        self.lm_head.forward(x)
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
            .map(Tensor::<B, 1, Int>::from)
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

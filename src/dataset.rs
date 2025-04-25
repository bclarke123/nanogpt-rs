use burn::data::dataset::Dataset;
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rngs::ThreadRng,
};

use crate::bigram::TrainingItem;

pub fn unique_chars(s: &str) -> Vec<char> {
    let mut chars = s.chars().collect::<Vec<_>>();
    chars.sort();
    chars.dedup();
    chars
}

fn stoi(c: char, vocab: &[char]) -> i32 {
    vocab
        .iter()
        .position(|&v| v == c)
        .unwrap_or_else(|| panic!("Character {} not found in vocabulary", c)) as i32
}

fn itos(index: i32, vocab: &[char]) -> char {
    vocab
        .get(index as usize)
        .copied()
        .unwrap_or_else(|| panic!("Index {} out of bounds for vocabulary", index))
}

pub fn encode(s: &str, vocab: &[char]) -> Vec<i32> {
    s.chars().map(|c| stoi(c, vocab)).collect::<Vec<_>>()
}

pub fn decode(indices: &[i32], vocab: &[char]) -> String {
    indices.iter().map(|&index| itos(index, vocab)).collect()
}

pub fn multinomial_distrib(input: &[f32], rng: &mut ThreadRng) -> usize {
    let dist = WeightedIndex::new(input).unwrap();
    dist.sample(rng)
}

pub struct TrainingDataset {
    pub block_size: usize,
    pub content: Vec<i32>,
}

impl TrainingDataset {
    pub fn new(dataset: &str, block_size: usize, vocab: &[char]) -> Self {
        let content = encode(dataset, vocab);

        Self {
            block_size,
            content,
        }
    }
}

impl Dataset<TrainingItem> for TrainingDataset {
    fn get(&self, index: usize) -> Option<TrainingItem> {
        Some(TrainingItem {
            context: self.content[index..index + self.block_size].to_vec(),
            target: self.content[index + 1..index + self.block_size + 1].to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.content.len() - self.block_size
    }
}

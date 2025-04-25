use burn::data::dataset::Dataset;

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

pub fn sample_distribution(distribution: &[f32]) -> usize {
    let mut cdf = Vec::with_capacity(distribution.len());
    let mut sum = 0.0;
    for &prob in distribution.iter() {
        sum += prob;
        cdf.push(sum);
    }

    // Normalize the CDF if necessary
    let cdf_last = *cdf.last().unwrap();
    if cdf_last != 1.0 {
        for cdf_val in cdf.iter_mut() {
            *cdf_val /= cdf_last;
        }
    }

    let random_value = rand::random_range(0f32..1f32);

    // Step 4: Find the index in the CDF
    cdf.iter()
        .position(|&x| x >= random_value)
        .unwrap_or_else(|| cdf.len() - 1)
}

pub struct TrainingDataset {
    pub block_size: usize,
    pub content: Vec<i32>,
}

impl TrainingDataset {
    pub fn new(dataset: &str, vocab: &[char]) -> Self {
        let content = encode(dataset, vocab);

        Self {
            block_size: 8,
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

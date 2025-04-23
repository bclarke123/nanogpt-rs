use burn::data::dataset::Dataset;

pub fn unique_chars(s: &str) -> Vec<char> {
    let mut chars = s.chars().collect::<Vec<_>>();
    chars.sort();
    chars.dedup();
    chars
}

fn stoi(c: char, vocab: &Vec<char>) -> u32 {
    vocab
        .iter()
        .position(|&v| v == c)
        .unwrap_or_else(|| panic!("Character {} not found in vocabulary", c)) as u32
}

fn itos(index: u32, vocab: &Vec<char>) -> char {
    vocab
        .get(index as usize)
        .copied()
        .unwrap_or_else(|| panic!("Index {} out of bounds for vocabulary", index))
}

fn encode(s: &str, vocab: &Vec<char>) -> Vec<u32> {
    s.chars().map(|c| stoi(c, vocab)).collect::<Vec<_>>()
}

fn decode(indices: &[u32], vocab: &Vec<char>) -> String {
    indices.iter().map(|&index| itos(index, vocab)).collect()
}

#[derive(Clone, Debug)]
pub struct TrainingItem {
    pub context: Vec<u32>,
    pub target: Vec<u32>,
}

pub struct TrainingDataset {
    pub block_size: usize,
    pub content: Vec<u32>,
}

impl TrainingDataset {
    pub fn new(dataset: &str, vocab: &Vec<char>) -> Self {
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

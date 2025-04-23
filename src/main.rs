use core::str;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use burn::{
    backend::Wgpu,
    tensor::{Device, Int, Tensor, TensorData, TensorKind},
};
use reqwest::Client;

type Backend = Wgpu;

async fn load_dataset(url: &str) -> Result<String> {
    static CLIENT: LazyLock<Client> = LazyLock::new(Client::new);

    CLIENT
        .get(url)
        .send()
        .await
        .context("Failed to send request")?
        .text()
        .await
        .context("Failed to read response text")
}

fn unique_chars(s: &str) -> Vec<char> {
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

fn get_batch(
    data: &[u32],
    block_size: usize,
    batch_size: usize,
) -> (Tensor<Backend, 2, Int>, Tensor<Backend, 2, Int>) {
    let idx = rand::random_range(0..data.len() - block_size);

    let x = Tensor::stack::<2>(
        (0..batch_size)
            .map(|_| {
                Tensor::<Backend, 1, Int>::from(TensorData::from(&data[idx..idx + block_size]))
            })
            .collect::<Vec<_>>(),
        0,
    );

    let y = Tensor::stack::<2>(
        (0..batch_size)
            .map(|_| {
                Tensor::<Backend, 1, Int>::from(TensorData::from(
                    &data[idx + 1..idx + block_size + 1],
                ))
            })
            .collect::<Vec<_>>(),
        0,
    );

    (x, y)
}

#[tokio::main]
async fn main() -> Result<()> {
    let url =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

    let dataset = load_dataset(url).await?;
    let chars = unique_chars(&dataset);

    println!("Unique characters: {}", String::from_iter(chars.iter()));

    let encoded = encode("hii there", &chars);
    println!("Encoded: {:?}", encoded);

    let decoded = decode(&encoded, &chars);
    println!("Decoded: {}", decoded);

    let encoded = encode(&dataset, &chars);
    let len = encoded.len() / 10 * 9;

    let train_data = &encoded[..len];
    let val_data = &encoded[len..];

    println!(
        "Using {len} bytes to train, {} to validate",
        encoded.len() - len
    );

    let batch_size = 4;
    let block_size = 8;

    let (xb, yb) = get_batch(train_data, block_size, batch_size);

    println!("inputs: {:?} / {:?}", xb.shape(), xb.to_string());
    println!("targets: {:?} / {:?}", yb.shape(), yb.to_string());

    // println!("{:?}", &encoded[..=block_size]);

    // let device = Default::default();
    // let tensor_data = TensorData::new(encoded[..len].to_vec(), [len; 1]);

    // let tensor: Tensor<Backend, 1, Int> =
    //     Tensor::<Backend, 1, Int>::from_data(tensor_data, &device);

    Ok(())
}

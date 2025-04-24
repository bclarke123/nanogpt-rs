use std::{env, fs};

use anyhow::Result;
use bigram::BigramModelConfig;
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    optim::AdamWConfig,
    prelude::*,
};
use dataset::unique_chars;
use ops::{TrainingConfig, generate, train};

mod bigram;
mod dataset;
mod ops;

const DATASET: &str = include_str!("tiny-shakespeare.txt");
const OUTPUT_DIR: &str = "output";

type NGBackend = Wgpu;
type NGAutodiffBackend = Autodiff<NGBackend>;

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

    let args = env::args();
    let op = if args.len() > 1 {
        args.collect::<Vec<_>>()[1].clone()
    } else {
        "generate".to_string()
    };

    if "train" == op {
        train::<NGAutodiffBackend>(
            OUTPUT_DIR,
            config,
            device,
            training_data,
            valid_data,
            &vocab,
        )?;
    } else {
        generate::<NGAutodiffBackend>(OUTPUT_DIR, device, &vocab, 500)?;
    }

    Ok(())
}

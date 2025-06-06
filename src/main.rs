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

const DATASET: &str = include_str!("data/tiny-shakespeare.txt");
const OUTPUT_DIR: &str = "output";

type NGBackend = Wgpu;
type NGAutodiffBackend = Autodiff<NGBackend>;

fn main() -> Result<()> {
    let training_len = DATASET.len() * 9 / 10;
    let training_data = &DATASET[..training_len];
    let valid_data = &DATASET[training_len + 1..];
    let vocab = unique_chars(DATASET);

    let device = WgpuDevice::DefaultDevice;
    let config_file = format!("{}/config.json", OUTPUT_DIR);

    fs::create_dir_all(OUTPUT_DIR)?;

    let config = if let Ok(cfg) = TrainingConfig::load(&config_file) {
        cfg
    } else {
        let cfg = TrainingConfig::new(
            BigramModelConfig::new().with_vocab_size(vocab.len()),
            AdamWConfig::new(),
        );

        cfg.save(&config_file)?;

        cfg
    };

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
        generate::<NGBackend>(OUTPUT_DIR, device, &vocab, 10000)?;
    }

    Ok(())
}

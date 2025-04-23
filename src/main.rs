use std::sync::LazyLock;

use anyhow::{Context, Result};
use reqwest::Client;

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

#[tokio::main]
async fn main() -> Result<()> {
    let url =
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

    let dataset = load_dataset(url).await?;

    let first_line = dataset.lines().next().context("Dataset is empty")?;

    println!("First line of the dataset: {}", first_line);

    Ok(())
}

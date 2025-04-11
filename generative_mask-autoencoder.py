import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from tqdm import tqdm
import transformers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log transformers version
logging.info(f"Using transformers version: {transformers.__version__}")

# Configuration
config = {
    'bitext_csv': os.path.expanduser("/home/curleyd/GitHub/New/bitext.csv"),
    'output_csv': os.path.expanduser("/home/curleyd/GitHub/New/anomalous_with_alternatives.csv"),
    'zrl_col': 4,  # Zero-resource language (column 5, 0-based index)
    'english_col': 5,  # English (column 6)
    'zrl_lang_code': 'de_DE',  # Placeholder for ZRL; adjust if known
    'max_length': 50,
    'embedding_dim': 128,
    'latent_dim': 128,
    'mae_epochs': 10,
    'batch_size': 32,
    'val_split': 0.2,
    'learning_rate': 1e-4,
    'mask_prob': 0.3,
    'error_threshold': 0.15,
    'early_stopping_patience': 5,
    'translation_epochs': 3,
    'translation_batch_size': 16,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Load bitexts.csv
try:
    df = pd.read_csv(config['bitext_csv'], header=None)
    df['id'] = df.iloc[:, 0].astype(str)
except FileNotFoundError:
    logging.error(f"Could not find {config['bitext_csv']}")
    exit(1)

# Initialize BERT tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config['vocab_size'] = tokenizer.vocab_size
except Exception as e:
    logging.error(f"Failed to load BERT tokenizer: {e}")
    exit(1)

# Dataset for MAE
class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, mask_prob):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Ensure text is string
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        masked_input = input_ids.clone()
        mask = torch.rand(input_ids.shape) < self.mask_prob
        masked_input[mask] = self.mask_token_id
        return masked_input, input_ids

    def __getitems__(self, indices):
        return [self[idx] for idx in indices]  # Handle batch requests

# Masked Autoencoder model
class MaskedAutoencoder(nn.Module):
    def __init__(self, vocab_size, max_length, embedding_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=latent_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=latent_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x).transpose(0, 1)
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded, encoded)
        logits = self.output_layer(decoded).transpose(0, 1)
        return logits

# Compute reconstruction error
def compute_error(model, tokenizer, text, max_length, device):
    try:
        encoding = tokenizer(
            str(text), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
            probs = F.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probs, dim=-1)
        error = torch.mean((input_ids != predicted_ids).float()).item()
        return error
    except Exception as e:
        logging.warning(f"Error computing reconstruction for text: {e}")
        return float('inf')  # Skip problematic texts

# Train MAE
def train_mae(model, train_loader, val_loader, criterion, optimizer, config):
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(config['mae_epochs']):
        model.train()
        train_loss = 0
        for masked_input, target in tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{config['mae_epochs']}"):
            masked_input, target = masked_input.to(config['device']), target.to(config['device'])
            logits = model(masked_input)
            loss = criterion(logits.reshape(-1, config['vocab_size']), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_avg = train_loss / len(train_loader)
        logging.info(f"MAE Epoch {epoch+1} - Train Loss: {train_loss_avg:.6f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for masked_input, target in val_loader:
                masked_input, target = masked_input.to(config['device']), target.to(config['device'])
                logits = model(masked_input)
                loss = criterion(logits.reshape(-1, config['vocab_size']), target.reshape(-1))
                val_loss += loss.item()
        val_loss_avg = val_loss / len(val_loader)
        logging.info(f"MAE Epoch {epoch+1} - Validation Loss: {val_loss_avg:.6f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), "best_mae.pth")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logging.info("Early stopping triggered for MAE.")
                break

# Identify anomalous sentences
def get_anomalous_sentences(model, tokenizer, texts, ids, max_length, device, threshold):
    anomalous = []
    model.eval()
    for idx, text in tqdm(enumerate(texts), total=len(texts), desc="Identifying anomalies"):
        error = compute_error(model, tokenizer, text, max_length, device)
        if error > threshold and error != float('inf'):
            anomalous.append((ids[idx], text, error))
    return sorted(anomalous, key=lambda x: x[2], reverse=True)  # Sort by error descending

# Fine-tune mBART for English to ZRL translation
def fine_tune_mbart(df, config):
    try:
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        tokenizer.tgt_lang = config['zrl_lang_code']
    except Exception as e:
        logging.error(f"Failed to load mBART model: {e}")
        exit(1)

    # Prepare dataset
    parallel_data = [
        {"src": str(row[config['english_col']]), "tgt": str(row[config['zrl_col']])}
        for _, row in df.iterrows()
        if pd.notna(row[config['english_col']]) and pd.notna(row[config['zrl_col']])
        and isinstance(row[config['english_col']], str) and isinstance(row[config['zrl_col']], str)
        and row[config['english_col']].strip() and row[config['zrl_col']].strip()
    ]
    
    if not parallel_data:
        logging.error("No valid parallel data found for fine-tuning.")
        exit(1)
        
    dataset = Dataset.from_list(parallel_data)
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Dataset sample: {dataset[:2]}")

    def tokenize_function(examples):
        src = examples["src"]
        tgt = examples["tgt"]
        model_inputs = tokenizer(
            src,
            text_target=tgt,
            padding="max_length",
            truncation=True,
            max_length=config['max_length']
        )
        return model_inputs

    # Test tokenization
    test_batch = {"src": dataset[:2]["src"], "tgt": dataset[:2]["tgt"]}
    try:
        test_output = tokenize_function(test_batch)
        logging.info("Tokenization test successful.")
    except Exception as e:
        logging.error(f"Tokenization test failed: {e}")
        exit(1)

    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
    except Exception as e:
        logging.error(f"Failed to tokenize dataset: {e}")
        exit(1)

    # Split dataset
    train_size = int((1 - config['val_split']) * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./mbart_results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=config['translation_batch_size'],
        per_device_eval_batch_size=config['translation_batch_size'],
        num_train_epochs=config['translation_epochs'],
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Failed to fine-tune mBART: {e}")
        exit(1)

    # Save model
    model.save_pretrained("fine_tuned_mbart")
    tokenizer.save_pretrained("fine_tuned_mbart")
    return model, tokenizer

# Generate alternative rendering
def generate_alternative(english_text, model, tokenizer, config):
    try:
        model.eval()
        inputs = tokenizer(
            str(english_text), return_tensors="pt", padding=True, 
            truncation=True, max_length=config['max_length']
        ).to(config['device'])
        translated_ids = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id[config['zrl_lang_code']],
            max_length=config['max_length']
        )
        alternative_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return alternative_text
    except Exception as e:
        logging.warning(f"Failed to generate alternative: {e}")
        return None

# Main function
def main():
    # Extract ZRL texts for MAE training
    zrl_texts = df.iloc[:, config['zrl_col']].astype(str).tolist()
    ids = df['id'].tolist()

    # Train MAE
    try:
        dataset = MaskedTextDataset(zrl_texts, tokenizer, config['max_length'], config['mask_prob'])
        train_size = int((1 - config['val_split']) * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])
    except Exception as e:
        logging.error(f"Failed to create MAE dataset: {e}")
        exit(1)

    mae_model = MaskedAutoencoder(
        config['vocab_size'], config['max_length'], config['embedding_dim'], config['latent_dim']
    ).to(config['device'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(mae_model.parameters(), lr=config['learning_rate'])
    train_mae(mae_model, train_loader, val_loader, criterion, optimizer, config)

    try:
        mae_model.load_state_dict(torch.load("best_mae.pth"))
    except FileNotFoundError:
        logging.error("Best MAE model not found")
        exit(1)

    # Identify anomalous sentences
    anomalous_sentences = get_anomalous_sentences(
        mae_model, tokenizer, zrl_texts, ids, config['max_length'], config['device'], config['error_threshold']
    )
    logging.info(f"Found {len(anomalous_sentences)} anomalous sentences.")

    # Fine-tune mBART
    mbart_model, mbart_tokenizer = fine_tune_mbart(df, config)
    mbart_model.to(config['device'])

    # Generate alternative renderings
    results = []
    for sent_id, original_text, error in tqdm(anomalous_sentences, desc="Generating alternatives"):
        try:
            # Find the corresponding English text
            row = df[df['id'] == sent_id]
            if row.empty:
                logging.warning(f"No row found for ID {sent_id}")
                continue
            english_text = row.iloc[0, config['english_col']]
            if pd.isna(english_text):
                logging.warning(f"No English text for ID {sent_id}")
                continue
            # Generate alternative ZRL text
            alternative_text = generate_alternative(english_text, mbart_model, mbart_tokenizer, config)
            if alternative_text is None:
                continue
            # Compute error for alternative
            alt_error = compute_error(mae_model, tokenizer, alternative_text, config['max_length'], config['device'])
            if alt_error == float('inf'):
                continue
            results.append({
                'id': sent_id,
                'text_original': original_text,
                'text_alternative': alternative_text,
                'error_original': error,
                'error_alternative': alt_error
            })
        except Exception as e:
            logging.warning(f"Failed to process ID {sent_id}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(config['output_csv'], index=False)
        logging.info(f"Saved results to {config['output_csv']}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        exit(1)

if __name__ == "__main__":
    main()

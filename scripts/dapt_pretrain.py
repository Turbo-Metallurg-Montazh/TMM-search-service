# dapt_pretrain.py

import os
import glob
import math
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import pandas as pd  # for Excel support

# ---------------- BASE CONFIG ----------------

BASE_MODEL_NAME = "xlm-roberta-base"    # or other encoder model
DAPT_OUTPUT_DIR = "dapt_model"
DAPT_CORPUS_DIR = "dapt"

MAX_LEN = 128
LR = 5e-5
MLM_PROB = 0.15

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ---------------- FILE LOADING ----------------

def extract_text_from_excel(path: str) -> List[str]:
    """Extract all text from all sheets & cells in an Excel file."""
    texts = []
    try:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, dtype=str)
            all_text = []
            for col in df.columns:
                col_data = df[col].dropna().astype(str).tolist()
                all_text.extend(col_data)
            joined = "\n".join(all_text).strip()
            if joined:
                texts.append(joined)
    except Exception as e:
        print(f"Failed to read Excel file {path}: {e}")
    return texts


def load_corpus_texts(corpus_dir: str) -> List[str]:
    """Read all text from text files and Excel files."""
    texts = []
    pattern = os.path.join(corpus_dir, "**", "*")

    for path in glob.glob(pattern, recursive=True):
        if os.path.isdir(path):
            continue

        lower = path.lower()
        try:
            # Excel files
            if lower.endswith(".xlsx") or lower.endswith(".xls"):
                excel_texts = extract_text_from_excel(path)
                texts.extend(excel_texts)
                continue

            # Plain text
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    texts.append(txt)

        except UnicodeDecodeError:
            # Non-text file, skip
            continue
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")
            continue

    return texts


def estimate_tokens(texts: List[str]) -> int:
    """Roughly estimate number of tokens in corpus from character count."""
    total_chars = sum(len(t) for t in texts)
    # crude approx: 1 token ~ 4 characters
    return max(1, total_chars // 4)


def choose_hyperparams(num_tokens: int):
    """Pick BATCH_SIZE and EPOCHS based on total token count."""
    if num_tokens < 200_000:
        batch_size = 16
        epochs = 5
    elif num_tokens < 1_000_000:
        batch_size = 32
        epochs = 3
    elif num_tokens < 5_000_000:
        batch_size = 32
        epochs = 2
    else:
        batch_size = 64
        epochs = 1

    return batch_size, epochs


# ---------------- DATASET ----------------

class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = list(texts)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ---------------- MASKING ----------------

def mask_tokens(input_ids: torch.Tensor, tokenizer, mlm_probability: float = 0.15):
    """Standard MLM masking."""
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)

    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tok in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
        if tok is not None:
            special_tokens_mask |= (input_ids == tok)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 80% -> [MASK]
    replace_prob = torch.full(labels.shape, 0.8, device=input_ids.device)
    indices_replaced = masked_indices & (torch.bernoulli(replace_prob).bool())
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not have a [MASK] token.")
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    random_prob = torch.full(labels.shape, 0.5, device=input_ids.device)
    indices_random = masked_indices & ~indices_replaced & (torch.bernoulli(random_prob).bool())
    random_words = torch.randint(len(tokenizer), labels.shape, device=input_ids.device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> unchanged

    return input_ids, labels


# ---------------- TRAINING ----------------

def main():
    print(f"Loading tokenizer and base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE)

    print(f"Reading domain text from ./{DAPT_CORPUS_DIR} ...")
    texts = load_corpus_texts(DAPT_CORPUS_DIR)

    if not texts:
        raise RuntimeError("No valid text found in ./dapt folder.")

    num_docs = len(texts)
    num_tokens = estimate_tokens(texts)

    batch_size, epochs = choose_hyperparams(num_tokens)

    print(f"Found {num_docs} documents in dapt/")
    print(f"Estimated total tokens: {num_tokens:,}")
    print(f"Chosen hyperparameters:")
    print(f"  BATCH_SIZE = {batch_size}")
    print(f"  EPOCHS     = {epochs}")
    print(f"  LR         = {LR}")
    print(f"  MAX_LEN    = {MAX_LEN}")
    print(f"Training on device: {DEVICE}\n")

    dataset = MLMDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batch in tqdm(dataloader, desc=f"EPOCH {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            masked_ids, labels = mask_tokens(input_ids, tokenizer, mlm_probability=MLM_PROB)

            outputs = model(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} MLM loss = {avg_loss:.4f}")

    os.makedirs(DAPT_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(DAPT_OUTPUT_DIR)
    tokenizer.save_pretrained(DAPT_OUTPUT_DIR)
    print(f"\nDAPT model saved to: {DAPT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

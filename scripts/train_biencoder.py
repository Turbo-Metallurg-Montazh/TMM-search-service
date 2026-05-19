import os
import math
import re
import random

# ============================================================
# Threading / CPU settings - must be BEFORE importing torch
# ============================================================
NUM_THREADS = 14

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_PROC_BIND"] = "TRUE"
os.environ["OMP_PLACES"] = "cores"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================
# Model + training config
# ============================================================

BASE_MODEL_NAME = "xlm-roberta-base"
DAPT_DIR = "dapt_model"

if os.path.isdir(DAPT_DIR):
    print(f"Using domain-adapted model from {DAPT_DIR}")
    MODEL_NAME = DAPT_DIR
else:
    print(f"Using base model {BASE_MODEL_NAME}")
    MODEL_NAME = BASE_MODEL_NAME

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
VAL_SIZE = 0.15

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)

print("DEVICE:", DEVICE)
print("Torch compute threads:", torch.get_num_threads())

# ============================================================
# Parameter parsing (same spirit as in match script)
# ============================================================

SOCKET_RE = re.compile(r'\b(GU10|G[0-9]{2}|E[0-9]{2})\b', re.IGNORECASE)
TUBE_RE   = re.compile(r'\bT\s*([0-9])\b', re.IGNORECASE)
IP_RE     = re.compile(r'\bIP\s*([0-9]{2})\b', re.IGNORECASE)
VOLT_RE   = re.compile(r'(\d{2,4})\s*[VВ]\b', re.IGNORECASE)
AMP_RE    = re.compile(r'(\d{1,3})\s*[AА]\b', re.IGNORECASE)
POWER_RE  = re.compile(r'(\d{1,3})\s*(Вт|W)\b', re.IGNORECASE)
LENGTH_RE = re.compile(r'(\d{3,4})\s*(мм|mm)\b', re.IGNORECASE)
KELVIN_RE = re.compile(r'(\d{3,5})\s*[KК]\b', re.IGNORECASE)


def extract_params(text: str) -> dict:
    """Extract structured parameters from a B-string."""
    t = text.replace(",", " ")
    params = {}

    m = SOCKET_RE.search(t)
    if m:
        params["socket"] = m.group(1).upper()

    m = TUBE_RE.search(t)
    if m:
        try:
            params["tube"] = int(m.group(1))
        except ValueError:
            pass

    m = IP_RE.search(t)
    if m:
        try:
            params["ip"] = int(m.group(1))
        except ValueError:
            pass

    m = VOLT_RE.search(t)
    if m:
        try:
            params["volt"] = int(m.group(1))
        except ValueError:
            pass

    m = AMP_RE.search(t)
    if m:
        try:
            params["amp"] = int(m.group(1))
        except ValueError:
            pass

    m = POWER_RE.search(t)
    if m:
        try:
            params["watt"] = int(m.group(1))
        except ValueError:
            pass

    m = LENGTH_RE.search(t)
    if m:
        try:
            params["len"] = int(m.group(1))
        except ValueError:
            pass

    m = KELVIN_RE.search(t)
    if m:
        try:
            params["kelvin"] = int(m.group(1))
        except ValueError:
            pass

    return params


def relative_diff(a: float, b: float) -> float:
    m = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / m


def parameter_mismatch_penalty(p1: dict, p2: dict) -> float:
    """
    Same style as in match script: bigger value = more parameter mismatch.
    Used only to *select* negatives, not in scoring.
    """
    penalty = 0.0

    # Socket / base type
    if "socket" in p1 and "socket" in p2 and p1["socket"] != p2["socket"]:
        penalty += 0.7

    # Tube
    if "tube" in p1 and "tube" in p2 and p1["tube"] != p2["tube"]:
        penalty += 0.5

    # IP
    if "ip" in p1 and "ip" in p2 and p1["ip"] != p2["ip"]:
        penalty += 0.5

    # Voltage
    if "volt" in p1 and "volt" in p2:
        if abs(p1["volt"] - p2["volt"]) >= 50:
            penalty += 0.4

    # Current
    if "amp" in p1 and "amp" in p2:
        rdiff = relative_diff(p1["amp"], p2["amp"])
        if rdiff >= 0.25:
            penalty += 0.3

    # Power
    if "watt" in p1 and "watt" in p2:
        rdiff = relative_diff(p1["watt"], p2["watt"])
        if rdiff >= 0.5:
            penalty += 0.3
        elif rdiff >= 0.2:
            penalty += 0.15

    # Length
    if "len" in p1 and "len" in p2:
        rdiff = relative_diff(p1["len"], p2["len"])
        if rdiff >= 0.3:
            penalty += 0.3
        elif rdiff >= 0.15:
            penalty += 0.15

    # Color temperature
    if "kelvin" in p1 and "kelvin" in p2:
        diff_k = abs(p1["kelvin"] - p2["kelvin"])
        if diff_k >= 2000:
            penalty += 0.3
        elif diff_k >= 1000:
            penalty += 0.15

    return min(penalty, 1.5)


def build_negatives(texts_b):
    """
    For each B+, choose a B- from all B's that has a large parameter mismatch.
    If no good candidate, leave empty string (no explicit negative for that row).
    """
    n = len(texts_b)
    params_all = [extract_params(t) for t in texts_b]
    neg_texts = []

    NEG_THRESHOLD = 0.7  # require at least this penalty to consider it "good" negative

    for i in range(n):
        p_pos = params_all[i]
        best_j = None
        best_pen = 0.0

        for j in range(n):
            if j == i:
                continue
            pen = parameter_mismatch_penalty(p_pos, params_all[j])
            if pen > best_pen:
                best_pen = pen
                best_j = j

        if best_j is not None and best_pen >= NEG_THRESHOLD:
            neg_texts.append(texts_b[best_j])
        else:
            neg_texts.append("")  # no explicit negative

    num_with_neg = sum(1 for t in neg_texts if t.strip() != "")
    print(f"Built explicit negatives for {num_with_neg} of {n} pairs "
          f"({100.0 * num_with_neg / max(1, n):.1f}%).")
    return neg_texts

# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):
    def __init__(self, texts_a, texts_b, tokenizer, texts_b_neg=None):
        self.texts_a = list(texts_a)
        self.texts_b = list(texts_b)
        self.texts_b_neg = list(texts_b_neg) if texts_b_neg is not None else None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts_a)

    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
        )

    def __getitem__(self, idx):
        a = self.texts_a[idx]
        b = self.texts_b[idx]

        input_ids_a, attn_a = self._encode(a)
        input_ids_b, attn_b = self._encode(b)

        item = {
            "input_ids_a": input_ids_a,
            "attention_mask_a": attn_a,
            "input_ids_b": input_ids_b,
            "attention_mask_b": attn_b,
        }

        if self.texts_b_neg is not None:
            b_neg = self.texts_b_neg[idx]
            if b_neg.strip():
                input_ids_bn, attn_bn = self._encode(b_neg)
                has_neg = 1.0
            else:
                # encode empty string just to fill tensors, but mark has_neg = 0
                input_ids_bn, attn_bn = self._encode("")
                has_neg = 0.0

            item["input_ids_b_neg"] = input_ids_bn
            item["attention_mask_b_neg"] = attn_bn
            item["has_neg"] = torch.tensor(has_neg, dtype=torch.float32)

        return item

# ============================================================
# BiEncoder model
# ============================================================

class BiEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean = summed / counts
        return mean  # (B, H)

    def forward(self, batch):
        emb_a = self.encode(batch["input_ids_a"], batch["attention_mask_a"])
        emb_b = self.encode(batch["input_ids_b"], batch["attention_mask_b"])
        return emb_a, emb_b

# ============================================================
# Contrastive loss + explicit negatives
# ============================================================

def contrastive_loss_with_explicit(
    emb_a_pos,
    emb_b_pos,
    emb_b_neg=None,
    has_neg=None,
    temperature=0.05,
    margin=0.2,
    lambda_neg=1.0,
):
    """
    InfoNCE + optional margin loss with explicit negatives.
    If emb_b_neg or has_neg is None, behaves exactly like the old contrastive loss.
    """
    # base InfoNCE
    emb_a = nn.functional.normalize(emb_a_pos, p=2, dim=1)
    emb_b = nn.functional.normalize(emb_b_pos, p=2, dim=1)

    logits = torch.matmul(emb_a, emb_b.t()) / temperature  # (B, B)
    labels = torch.arange(emb_a.size(0), device=emb_a.device)

    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_j = nn.functional.cross_entropy(logits.t(), labels)
    info_nce_loss = (loss_i + loss_j) / 2.0

    # no explicit negatives → just InfoNCE
    if emb_b_neg is None or has_neg is None:
        return info_nce_loss

    mask = has_neg > 0.5
    if mask.sum() == 0:
        return info_nce_loss

    emb_bneg = nn.functional.normalize(emb_b_neg, p=2, dim=1)

    emb_a_sel = emb_a[mask]
    emb_b_sel = emb_b[mask]
    emb_bneg_sel = emb_bneg[mask]

    sim_pos = (emb_a_sel * emb_b_sel).sum(dim=1)
    sim_neg = (emb_a_sel * emb_bneg_sel).sum(dim=1)

    margin_loss = torch.relu(margin + sim_neg - sim_pos).mean()

    return info_nce_loss + lambda_neg * margin_loss

# ============================================================
# Main training routine
# ============================================================

def main():
    # ----- Load data -----
    df = pd.read_excel("samplescleaned.xlsx")  # first col A, second col B

    texts_a = df.iloc[:, 0].astype(str).tolist()
    texts_b = df.iloc[:, 1].astype(str).tolist()

    # Build parameter-based explicit negatives on B side
    texts_b_neg = build_negatives(texts_b)

    # Train/val split (A, B, B_neg together)
    train_a, val_a, train_b, val_b, train_b_neg, val_b_neg = train_test_split(
        texts_a,
        texts_b,
        texts_b_neg,
        test_size=VAL_SIZE,
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PairDataset(train_a, train_b, tokenizer, texts_b_neg=train_b_neg)
    val_ds   = PairDataset(val_a,   val_b,   tokenizer, texts_b_neg=val_b_neg)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    model = BiEncoder(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ----- Training loop -----
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()
            emb_a, emb_b = model(batch)

            if "input_ids_b_neg" in batch:
                emb_b_neg = model.encode(
                    batch["input_ids_b_neg"],
                    batch["attention_mask_b_neg"],
                )
                has_neg = batch["has_neg"]
                loss = contrastive_loss_with_explicit(
                    emb_a,
                    emb_b,
                    emb_b_neg=emb_b_neg,
                    has_neg=has_neg,
                    temperature=0.05,
                    margin=0.2,
                    lambda_neg=1.0,
                )
            else:
                loss = contrastive_loss_with_explicit(emb_a, emb_b)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # ----- Quick validation -----
        model.eval()
        with torch.no_grad():
            sims_pos = []
            sims_neg = []
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                emb_a, emb_b = model(batch)
                emb_a_n = nn.functional.normalize(emb_a, p=2, dim=1)
                emb_b_n = nn.functional.normalize(emb_b, p=2, dim=1)

                pos = (emb_a_n * emb_b_n).sum(dim=1)
                sims_pos.extend(pos.cpu().tolist())

                emb_b_shuf = emb_b_n[torch.randperm(emb_b_n.size(0))]
                neg = (emb_a_n * emb_b_shuf).sum(dim=1)
                sims_neg.extend(neg.cpu().tolist())

            mean_pos = sum(sims_pos) / len(sims_pos)
            mean_neg = sum(sims_neg) / len(sims_neg)
            print(
                f"  mean_pos_sim={mean_pos:.3f}, "
                f"mean_neg_sim={mean_neg:.3f}"
            )

    # ----- Save model & tokenizer -----
    model.encoder.save_pretrained("biencoder_model")
    tokenizer.save_pretrained("biencoder_model")
    print("Model saved to ./biencoder_model")


if __name__ == "__main__":
    main()

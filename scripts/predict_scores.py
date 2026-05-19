import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import Counter, defaultdict

# ====== CONFIG ======
MODEL_DIR = "biencoder_model"      # directory where train_biencoder.py saved the model
EXCEL_FILE = "samplescleaned.xlsx"          # your original file
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ====== LOAD MODEL & TOKENIZER ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
encoder = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
encoder.eval()


def encode_texts(texts):
    """Encode a list of texts into normalized embeddings."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = encoder(**enc)
        last_hidden = out.last_hidden_state         # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)                  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)    # (B, 1)
        mean = summed / counts
        return F.normalize(mean, p=2, dim=1)        # (B, H)


def main():
    # ====== LOAD DATA ======
    df = pd.read_excel(EXCEL_FILE)
    col_a = df.iloc[:, 0].astype(str).tolist()
    col_b = df.iloc[:, 1].astype(str).tolist()

    n_rows = len(df)
    n_samples = min(500, n_rows)

    # randomly choose row indices for evaluation
    indices = list(range(n_rows))
    random.shuffle(indices)
    eval_indices = indices[:n_samples]

    print(f"Evaluating on {len(eval_indices)} randomly chosen pairs...")

    # ====== PRECOMPUTE EMBEDDINGS FOR ALL CANDIDATE B TEXTS ======
    candidates_b = col_b
    emb_b_all = encode_texts(candidates_b)          # (N, H)
    emb_b_all_t = emb_b_all.t()                     # (H, N) for fast matmul

    # rank_counts[rank] = how many times correct B got this rank
    rank_counts = Counter()
    not_found = 0

    # store up to 10 examples per rank where correct B is NOT rank 1
    rank_examples = defaultdict(list)

    # ====== EVALUATION LOOP ======
    for idx in eval_indices:
        a_text = col_a[idx]
        true_b = col_b[idx]

        # encode this A
        emb_a = encode_texts([a_text])              # (1, H)

        # cosine similarity: since vectors are normalized, dot product = cosine
        sims = torch.matmul(emb_a, emb_b_all_t)     # (1, N)
        sims = sims.cpu().numpy()[0]                # (N,)

        # sort candidates by similarity descending
        sorted_indices = np.argsort(-sims)          # indices of candidates_b

        # find rank of the true B (first occurrence if duplicates)
        rank = None
        for r, j in enumerate(sorted_indices, start=1):
            if candidates_b[j] == true_b:
                rank = r
                break

        if rank is None:
            not_found += 1
            continue

        rank_counts[rank] += 1

        # if correct answer is not rank 1, store example (limit 10 per rank)
        if rank != 1 and len(rank_examples[rank]) < 10:
            top_idx = sorted_indices[0]
            top_b = candidates_b[top_idx]
            rank_examples[rank].append({
                "a_text": a_text,
                "true_b": true_b,
                "pred_b": top_b
            })

    # ====== PRINT RESULTS: RANK STATS ======
    print("\nRanking statistics (how often the correct B was at rank k):")
    for rank in sorted(rank_counts.keys()):
        count = rank_counts[rank]
        print(f"Rank {rank}: {count} times")

    total_eval = len(eval_indices)
    found = total_eval - not_found
    print(f"\nTotal evaluated pairs: {total_eval}")
    print(f"Pairs where correct B was found among candidates: {found}")
    if not_found > 0:
        print(f"WARNING: {not_found} pair(s) where the true B was not found in candidates_b (string mismatch).")

    # ====== PRINT WRONG FIRST-CHOICE EXAMPLES ======
    print("\nExamples where the first choice was WRONG (max 10 per rank):\n")
    for rank in sorted(rank_examples.keys()):
        examples = rank_examples[rank]
        print(f"--- Rank {rank} (showing {len(examples)} example(s)) ---")
        for i, ex in enumerate(examples, start=1):
            print(f"[{i}] A: {ex['a_text']}")
            print(f"     TRUE B : {ex['true_b']}")
            print(f"     PRED B : {ex['pred_b']}")
        print()

if __name__ == "__main__":
    main()

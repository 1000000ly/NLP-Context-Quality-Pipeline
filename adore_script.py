# ==============================
# ADORE-style training + evaluation (single cell, corrected & consistent)
# Follows the idea of ADORE (arXiv:2104.08051): train a query encoder while keeping doc encoder fixed.
# Assumes your dataset is HotpotQA-like: train/dev items have: _id, question, answer, supporting_facts, context
# and your corpus is a dict: {doc_id: {"title": ..., "text": ...}, ...}
# ==============================

print("Importing libraries...")
import os
import json
import random
import gc
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.functional import softplus
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import centralized utils
from utils.evaluation import exact_match_score, f1_score
from utils.prompts import create_prompt, extract_short_answer
from utils.data_loading import load_json

# ------------------------------
# Paths / config
# ------------------------------
print("Paths / config.........")

BASE_DIR = ""

TRAIN_DATA   = os.path.join(BASE_DIR, "data/train.json")
DEV_DATA     = os.path.join(BASE_DIR, "data/dev.json")
CORPUS_PATH  = os.path.join(BASE_DIR, "data/clean_wiki_corpus.json")

OUTPUT_DIR_ADORE = os.path.join(BASE_DIR, "output/adore")
RESULTS_DIR      = os.path.join(BASE_DIR, "results_new")
MODEL_SAVE_PATH  = os.path.join(OUTPUT_DIR_ADORE, "adore_model")

os.makedirs(OUTPUT_DIR_ADORE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_EPOCHS        = 3
BATCH_SIZE        = 64
LEARNING_RATE     = 2e-5
MAX_TRAIN_SAMPLES = None  # Set to None for all data, or a number to limit
LIMIT_DEV         = 1200 
MAX_K             = 5

# Retrieval/training knobs
HARD_NEG_POOL = 10     # consider top-N retrieved docs
MIN_NEG_RANK  = 3       # avoid top few (too easy / often positives)
DELTA_M_FLOOR = 0.05    # stabilize early training

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("✓ Setup")
print("Base directory:", BASE_DIR)
print("Device:", device)

# ------------------------------
# IO helper (save_json only, load_json imported from utils)
# ------------------------------
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------------------
# Load data (ALWAYS load corpus/train/dev so later code never references undefined vars)
# ------------------------------
print("\nLoading data...")
train_data = load_json(TRAIN_DATA)
dev_data   = load_json(DEV_DATA)
corpus_data = load_json(CORPUS_PATH)

if MAX_TRAIN_SAMPLES:
    train_data = train_data[:MAX_TRAIN_SAMPLES]
if LIMIT_DEV:
    dev_data = dev_data[:LIMIT_DEV]

print(f"✓ train: {len(train_data)} | dev: {len(dev_data)} | corpus: {len(corpus_data)}")

# ------------------------------
# Build corpus arrays
# ------------------------------
# corpus_data expected: dict(doc_id -> {title, text})
corpus_ids, corpus_titles, corpus_texts = [], [], []
if isinstance(corpus_data, dict):
    for doc_id, doc in corpus_data.items():
        # robust keys
        title = doc.get("title", str(doc_id))
        text  = doc.get("text", "")
        corpus_ids.append(doc_id)
        corpus_titles.append(title)
        corpus_texts.append(text)
else:
    raise ValueError("CORPUS format unexpected. Expected a dict {doc_id: {title, text}}")

# title -> indices for positives (supporting_facts titles)
title_to_indices = defaultdict(list)
for idx, title in enumerate(corpus_titles):
    title_to_indices[title].append(idx)

print(f"✓ Corpus built: {len(corpus_texts)} documents")

# ------------------------------
# Load Contriever (doc encoder frozen) + initialize query encoder copy
# ------------------------------
CONTRIEVER_PATH = os.path.join(BASE_DIR, "models/contriever")
if os.path.exists(CONTRIEVER_PATH):
    print("\nLoading Contriever from local:", CONTRIEVER_PATH)
    doc_encoder = SentenceTransformer(CONTRIEVER_PATH)
    query_encoder = SentenceTransformer(CONTRIEVER_PATH)
else:
    print("\nLoading Contriever from HF: facebook/contriever (requires internet)")
    doc_encoder = SentenceTransformer("facebook/contriever")
    query_encoder = SentenceTransformer("facebook/contriever")

doc_encoder.to(device)
query_encoder.to(device)

# Freeze doc encoder
for p in doc_encoder.parameters():
    p.requires_grad = False
doc_encoder.eval()

print("✓ Encoders ready")
print("Doc max seq len:", doc_encoder.max_seq_length, "| Query max seq len:", query_encoder.max_seq_length)

# ------------------------------
# Precompute (or load) corpus embeddings with doc encoder
# ------------------------------
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "indices/corpus/embeddings.pt")
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

def l2_normalize(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

if os.path.exists(EMBEDDINGS_PATH):
    print("\nLoading precomputed corpus embeddings:", EMBEDDINGS_PATH)
    corpus_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
else:
    print("\nComputing corpus embeddings (doc encoder frozen)...")
    with torch.no_grad():
        corpus_embeddings = doc_encoder.encode(
            corpus_texts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=device
        )
    torch.save(corpus_embeddings, EMBEDDINGS_PATH)
    print("✓ Saved corpus embeddings:", EMBEDDINGS_PATH)

corpus_embeddings = corpus_embeddings.to(device)
corpus_embeddings = l2_normalize(corpus_embeddings)
print("✓ corpus_embeddings:", tuple(corpus_embeddings.shape))

# ------------------------------
# Dataset / Dataloader
# ------------------------------
class ADOREDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

train_loader = DataLoader(
    ADOREDataset(train_data),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ------------------------------
# Positives helper (title-based)
# ------------------------------
def get_positive_indices(example):
    """
    supporting_facts expected: list of [title, sent_idx] pairs.
    We map title -> corpus indices.
    """
    indices = []
    for fact in example.get("supporting_facts", []):
        if not fact:
            continue
        title = fact[0]
        if title in title_to_indices:
            indices.extend(title_to_indices[title])
    # unique
    return list(set(indices))

# ------------------------------
# Efficient cosine: query_emb (B,D) vs corpus (N,D)
# both normalized => cosine = dot
# ------------------------------
def score_against_corpus(q_embs, corpus_embs):
    # q_embs: (B,D), corpus: (N,D) -> (B,N)
    return q_embs @ corpus_embs.T

# ------------------------------
# Train ADORE-style LambdaLoss
# ------------------------------
optimizer = AdamW(query_encoder.parameters(), lr=LEARNING_RATE)
query_encoder.train()

training_losses = []
step_losses = []  # Track individual step losses for rolling average
rolling_avg_losses = []  # Track (step, avg_loss) every 1000 steps
global_step = 0
LOG_EVERY_N_STEPS = 1000  # Compute and log rolling average every N steps

print("\n==============================")
print("TRAINING (ADORE-style)")
print("==============================")

for epoch in range(NUM_EPOCHS):
    epoch_loss, steps = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch in pbar:
        queries = [ex.get("question", "") for ex in batch]

        # Forward with gradients (DO NOT use .encode here)
        features = query_encoder.tokenize(queries)
        features = {k: v.to(device) for k, v in features.items()}
        out = query_encoder(features)
        q_embs = out["sentence_embedding"]
        q_embs = l2_normalize(q_embs)

        aligned_q, pos_embs, neg_embs, delta_ms = [], [], [], []

        # For each query in batch, choose one pos and one hard neg
        # (hard neg from top HARD_NEG_POOL, excluding positives, and rank > MIN_NEG_RANK)
        with torch.no_grad():
            all_scores = score_against_corpus(q_embs, corpus_embeddings)  # (B,N)

        for i in range(q_embs.size(0)):
            positives = get_positive_indices(batch[i])
            if not positives:
                continue

            pos_idx = random.choice(positives)
            pos_score = all_scores[i, pos_idx].item()

            # ranks in top pool
            topk = min(HARD_NEG_POOL, corpus_embeddings.size(0))
            top_scores, top_indices = torch.topk(all_scores[i], k=topk, largest=True)

            # pos rank within entire corpus (approx using greater-than count; OK but O(N) if computed exactly)
            # We'll estimate pos_rank by locating in topk if present; otherwise fallback to MIN_NEG_RANK+1.
            pos_rank = None
            for r, idx in enumerate(top_indices.tolist(), start=1):
                if idx == pos_idx:
                    pos_rank = r
                    break
            if pos_rank is None:
                pos_rank = MIN_NEG_RANK + 1

            neg_candidates = []
            for r, idx in enumerate(top_indices.tolist(), start=1):
                if r <= MIN_NEG_RANK:
                    continue
                if idx in positives:
                    continue
                neg_candidates.append((idx, r))

            if not neg_candidates:
                continue

            neg_idx, neg_rank = random.choice(neg_candidates)

            delta_m = abs((1.0 / float(neg_rank)) - (1.0 / float(pos_rank)))
            if delta_m < DELTA_M_FLOOR:
                delta_m = DELTA_M_FLOOR

            aligned_q.append(q_embs[i])
            pos_embs.append(corpus_embeddings[pos_idx])
            neg_embs.append(corpus_embeddings[neg_idx])
            delta_ms.append(delta_m)

        if not aligned_q:
            continue

        aligned_q = torch.stack(aligned_q, dim=0)
        pos_embs   = torch.stack(pos_embs, dim=0)
        neg_embs   = torch.stack(neg_embs, dim=0)
        delta_ms   = torch.tensor(delta_ms, device=device, dtype=torch.float32)

        # since embeddings are normalized: cosine = dot
        pos_scores = (aligned_q * pos_embs).sum(dim=-1)
        neg_scores = (aligned_q * neg_embs).sum(dim=-1)

        # LambdaLoss style: delta_m * softplus(neg - pos)
        loss = (delta_ms * softplus(neg_scores - pos_scores)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        steps += 1
        global_step += 1
        step_losses.append(loss.item())  # Store every step loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", pairs=len(aligned_q), step=global_step)
        
        # Compute rolling average over last 1000 steps every 1000 steps
        if global_step % LOG_EVERY_N_STEPS == 0:
            recent_losses = step_losses[-LOG_EVERY_N_STEPS:]
            avg_loss_1000 = sum(recent_losses) / len(recent_losses)
            rolling_avg_losses.append((global_step, avg_loss_1000))

        if steps % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = epoch_loss / max(1, steps)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    query_encoder.save(MODEL_SAVE_PATH)
    print("✓ Saved ADORE query encoder:", MODEL_SAVE_PATH)

print("\n✓ ADORE training complete")

# Plot training loss (both epoch-level and rolling average)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Epoch-level loss
ax1.plot(list(range(1, NUM_EPOCHS + 1)), training_losses, marker="o", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Average Training Loss", fontsize=12)
ax1.set_title("ADORE Training Loss (Per Epoch)", fontsize=14)
ax1.grid(True, alpha=0.3)

# Rolling average loss (every 1000 steps)
if rolling_avg_losses:
    steps_list = [s for s, l in rolling_avg_losses]
    losses_list = [l for s, l in rolling_avg_losses]
    ax2.plot(steps_list, losses_list, marker="o", alpha=0.8, linewidth=2)
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Average Loss (Last 1000 Steps)", fontsize=12)
    ax2.set_title("ADORE Training Loss (Rolling Average)", fontsize=14)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR_ADORE, "training_loss.png")
plt.savefig(loss_plot_path, dpi=150)
plt.show()
print("✓ Saved loss plot:", loss_plot_path)

# Save loss data for future reference
loss_data = {
    "epoch_losses": training_losses,
    "step_losses": step_losses,
    "config": {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "log_every_n_steps": LOG_EVERY_N_STEPS
    }
}
loss_data_path = os.path.join(OUTPUT_DIR_ADORE, "training_losses.json")
save_json(loss_data, loss_data_path)
print("✓ Saved loss data:", loss_data_path)

# ------------------------------
# Retrieval: ADORE top-k on dev
# ------------------------------
print("\n==============================")
print("RETRIEVAL (ADORE encoder)")
print("==============================")

adore_query_encoder = SentenceTransformer(MODEL_SAVE_PATH).to(device)
adore_query_encoder.eval()

dev_queries = [x.get("question", "") for x in dev_data]
dev_ids     = [x.get("_id", "") for x in dev_data]

with torch.no_grad():
    dev_q_embs = adore_query_encoder.encode(
        dev_queries,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
dev_q_embs = l2_normalize(dev_q_embs)

retrieval_results = []
with torch.no_grad():
    for qid, qtext, qemb in tqdm(zip(dev_ids, dev_queries, dev_q_embs), total=len(dev_ids), desc="Retrieving top-k"):
        scores = (qemb.unsqueeze(0) @ corpus_embeddings.T).squeeze(0)  # (N,)
        top_idx = torch.topk(scores, k=MAX_K).indices.tolist()

        docs = []
        for r, idx in enumerate(top_idx, start=1):
            docs.append({
                "doc_id": corpus_ids[idx],
                "title": corpus_titles[idx],
                "text": corpus_texts[idx],
                "score": float(scores[idx].cpu()),
                "rank": r
            })
        retrieval_results.append({
            "query_id": qid,
            "query_text": qtext,
            "retrieved_docs": docs
        })

def save_top_k(retrieval_results, k, outdir, prefix="adore"):
    trimmed = []
    for item in retrieval_results:
        trimmed.append({
            "query_id": item["query_id"],
            "query_text": item["query_text"],
            "retrieved_docs": item["retrieved_docs"][:k]
        })
    path = os.path.join(outdir, f"{prefix}_top_{k}_retrieval.json")
    save_json(trimmed, path)
    print(f"✓ Saved {prefix} top-{k} retrieval:", path)

for k in [1, 3, 5]:
    save_top_k(retrieval_results, k, OUTPUT_DIR_ADORE, prefix="adore")

# ------------------------------
# Generation + EM/F1 (using imported utils functions)
# ------------------------------

# Load Flan-T5 (local preferred)
print("\nLoading Flan-T5...")
FLANT5_PATH = os.path.join(BASE_DIR, "models/flan-t5-base")
if os.path.exists(FLANT5_PATH):
    gen_tokenizer = AutoTokenizer.from_pretrained(FLANT5_PATH)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_PATH).to(device)
else:
    # will only work if you have internet
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
gen_model.eval()
print("✓ Flan-T5 loaded")

gold_answers = {x.get("_id", ""): x.get("answer", "") for x in dev_data}

def run_generation_and_eval(retrieval_json_path, out_prefix):
    retrieval = load_json(retrieval_json_path)

    print(f"\n--- ANSWER GENERATION ---")
    results = []
    
    # Generation phase
    for item in tqdm(retrieval, desc=f"Generating answers [{out_prefix}]"):
        qid = item["query_id"]
        q   = item["query_text"]
        contexts = [d["text"] for d in item["retrieved_docs"]]

        prompt = create_prompt(q, contexts)
        inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        with torch.no_grad():
            out = gen_model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        gen = gen_tokenizer.decode(out[0], skip_special_tokens=True)
        pred = extract_short_answer(gen)

        gold = gold_answers.get(qid, "")

        results.append({
            "query_id": qid,
            "question": q,
            "generated_answer": pred,
            "gold_answer": gold,
        })
    
    print(f"✓ Generated {len(results)} answers")
    
    # Evaluation phase
    print(f"\n--- EVALUATION ---")
    exact_matches = 0
    total_f1 = 0.0
    
    for item in tqdm(results, desc=f"Computing metrics [{out_prefix}]"):
        pred = item["generated_answer"]
        gold = item["gold_answer"]
        
        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)
        
        item["exact_match"] = em
        item["f1_score"] = f1
        
        exact_matches += em
        total_f1 += f1

    n = max(1, len(results))
    metrics = {
        "num_queries": n,
        "exact_match": exact_matches / n,
        "f1_score": total_f1 / n,
        "exact_matches_count": exact_matches
    }
    return results, metrics

# ADORE eval for k in {1,3,5}
k_values = [1, 3, 5]
adore_metrics_all = {}

for k in k_values:
    print(f"\n{'='*80}\nADORE TOP-{k}\n{'='*80}")
    adore_retr = os.path.join(OUTPUT_DIR_ADORE, f"adore_top_{k}_retrieval.json")
    adore_results, adore_metrics = run_generation_and_eval(adore_retr, out_prefix=f"adore_top_{k}")
    adore_metrics.update({"model": "adore", "k": k})

    save_json(adore_results, os.path.join(RESULTS_DIR, f"adore_top_{k}_answers.json"))
    save_json(adore_metrics, os.path.join(RESULTS_DIR, f"adore_top_{k}_metrics.json"))
    adore_metrics_all[k] = adore_metrics

    print(f"EM: {adore_metrics['exact_match']:.4f} ({adore_metrics['exact_matches_count']}/{adore_metrics['num_queries']})")
    print(f"F1: {adore_metrics['f1_score']:.4f}")

# ------------------------------
# Summary
# ------------------------------
print("\n" + "="*80)
print("ADORE EVALUATION SUMMARY")
print("="*80)
print(f"{'k':<5} {'EM':<10} {'F1':<10}")
print("-" * 80)
for k in k_values:
    m = adore_metrics_all[k]
    print(f"{k:<5} {m['exact_match']:<10.4f} {m['f1_score']:<10.4f}")

save_json(adore_metrics_all, os.path.join(RESULTS_DIR, "adore_summary.json"))
print("\n✓ Saved ADORE summary:", os.path.join(RESULTS_DIR, "adore_summary.json"))
print("\n✓ Done.")

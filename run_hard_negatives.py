"""
Step 4: Hard Negative Sampling (OFF-THE-SHELF RETRIEVER, e.g., Contriever)

Hard negatives = documents that rank high for the query (close in vector space),
but are NOT in the ground-truth supporting documents (supporting_facts titles).

We run generation with contexts:
  [top-k retrieved contexts] + [k_hard hard negatives taken from high-ranked retrieved docs not in GT]
and evaluate with EM/F1.

Inputs expected:
  - data/dev.json
  - results/top_1_retrieval.json
  - results/top_3_retrieval.json
  - results/top_5_retrieval.json

Each retrieval json item should look like:
{
  "query_id": "...",
  "query_text": "...",
  "retrieved_docs": [
      {"doc_id": "...", "title": "...", "text": "...", "score": ..., "rank": ...},
      ...
  ]
}

IMPORTANT:
- Loads Flan-T5 from local cache at: <BASE_DIR>/models/flan-t5-base
- Uses ONLY the retrieval ranking to select hard negatives (no ADORE training).
"""

import os, json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.evaluation import normalize_answer, exact_match_score, f1_score
from utils.prompts import create_prompt, extract_short_answer
from utils.data_loading import load_json

# -------------------------
# Config
# -------------------------
DATA_PATH   = "data/dev.json"
RESULTS_DIR = "results"
LIMIT       = 1200

CONFIGS = [
    {"k_relevant": 1, "k_hard": 1, "name": "1plus1hard"},
    {"k_relevant": 3, "k_hard": 3, "name": "3plus3hard"},
    {"k_relevant": 5, "k_hard": 5, "name": "5plus5hard"},
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# IO helpers
# -------------------------
def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -------------------------
# GT titles + Hard negatives
# -------------------------
def get_supporting_titles(dev_item) -> set[str]:
    """
    dev_item["supporting_facts"] typically: [[title, sent_idx], ...]
    Return set of titles.
    """
    titles = set()
    for fact in dev_item.get("supporting_facts", []):
        if isinstance(fact, (list, tuple)) and len(fact) >= 1:
            titles.add(fact[0])
    return titles

def select_hard_negatives_from_retrieval(
    retrieved_docs: list[dict],
    supporting_titles: set[str],
    k_rel: int,
    k_hard: int,
) -> tuple[list[str], list[str], int, int]:
    """
    Select contexts for hard negative evaluation.
    
    Strategy:
    1. Take first k_rel retrieved docs (as baseline would)
    2. From remaining docs, take k_hard that are NOT in supporting_titles
    
    Returns:
        relevant_contexts: First k_rel retrieved docs (baseline contexts)
        hard_negative_contexts: k_hard hard negatives from remaining docs
        num_correct_in_top_k: How many of first k_rel are actually correct (for analysis)
        num_hard_found: How many hard negatives were found
    """
    # Take first k_rel as baseline would (may or may not be correct)
    relevant_contexts = [d.get("text", "") for d in retrieved_docs[:k_rel]]
    relevant_contexts = [c for c in relevant_contexts if c]
    
    # Count how many are actually correct (for debugging)
    num_correct = sum(1 for d in retrieved_docs[:k_rel] 
                     if (d.get("title") or d.get("doc_id", "")) in supporting_titles)
    
    # From remaining docs, select hard negatives (high-ranked but not in GT)
    hard_negatives = []
    for d in retrieved_docs[k_rel:]:
        title = d.get("title") or d.get("doc_id", "")
        txt = d.get("text", "")
        if not txt:
            continue
            
        if title not in supporting_titles:
            hard_negatives.append(txt)
            if len(hard_negatives) >= k_hard:
                break
    
    return relevant_contexts, hard_negatives, num_correct, len(hard_negatives)

# -------------------------
# Main
# -------------------------
def main():
    print("=" * 80)
    print("STEP 4: HARD NEGATIVE SAMPLING (off-the-shelf retriever ranking)")
    print("=" * 80)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    BASE_DIR = ""

    FLANT5_PATH = os.path.join(BASE_DIR, "models", "flan-t5-base")
    if not os.path.exists(FLANT5_PATH):
        raise FileNotFoundError(
            f"Local Flan-T5 not found at: {FLANT5_PATH}\n"
            f"Put the downloaded model there (or update FLANT5_PATH)."
        )

    # load dev
    dev_path = os.path.join(BASE_DIR, DATA_PATH)
    dev_data = load_json(dev_path)[:LIMIT]
    print(f"✓ Loaded {len(dev_data)} dev questions from {dev_path}")

    # load generator (LOCAL)
    print("\nLoading Flan-T5 locally...")
    print("Path:", FLANT5_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FLANT5_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_PATH).to(device)
    model.eval()
    print("✓ Flan-T5 loaded")

    all_summary = {}

    for cfg in CONFIGS:
        k_rel = cfg["k_relevant"]
        k_hard = cfg["k_hard"]
        name = cfg["name"]

        # Always use top-10 retrieval for hard negatives selection
        retrieval_path = os.path.join(BASE_DIR, "output", "top_10_retrieval_results.json")
        
        if not os.path.exists(retrieval_path):
            print(f"⚠ Warning: {retrieval_path} not found")
            print(f"   Please run: python run_retrieval_top10.py")
            continue

        retrieval_data = load_json(retrieval_path)
        retr_by_qid = {x["query_id"]: x for x in retrieval_data}

        print(f"\n{'='*80}")
        print(f"CONFIG: {name}  (top-{k_rel} + {k_hard} hard)")
        print(f"Retrieval file: {retrieval_path}")
        print(f"{'='*80}")

        results = []
        em_count = 0
        f1_sum = 0.0
        skipped = 0
        
        # Debug counters
        total_hard_found = 0
        total_correct_in_baseline = 0
        queries_with_no_hard = 0
        queries_with_empty_supporting = 0

        # Print sample data for first query to debug
        first_sample_printed = False
        
        for item in tqdm(dev_data, desc=name):
            qid = item["_id"]
            if qid not in retr_by_qid:
                skipped += 1
                continue

            question = item.get("question", "")
            gold = item.get("answer", "")
            supporting_titles = get_supporting_titles(item)

            retrieved_docs = retr_by_qid[qid].get("retrieved_docs", [])
            
            if len(retrieved_docs) == 0:
                skipped += 1
                continue

            # Get baseline contexts + hard negatives
            relevant_contexts, hard_contexts, num_correct, num_hard = select_hard_negatives_from_retrieval(
                retrieved_docs=retrieved_docs,
                supporting_titles=supporting_titles,
                k_rel=k_rel,
                k_hard=k_hard,
            )
            
            # Debug first query
            if not first_sample_printed:
                print(f"\n🔍 SAMPLE DEBUG (Query: {qid}):")
                print(f"  Question: {question[:80]}...")
                print(f"  Gold answer: {gold}")
                print(f"  Supporting titles (GT): {supporting_titles}")
                print(f"  Retrieved {len(retrieved_docs)} docs total")
                print(f"  Using first {k_rel} retrieved (baseline): {num_correct}/{k_rel} are correct")
                print(f"  Adding {len(hard_contexts)} hard negatives from remaining docs")
                print(f"  Total contexts: {len(relevant_contexts)} baseline + {len(hard_contexts)} hard")
                print(f"  Top-5 retrieved titles: {[d.get('title', 'NO_TITLE')[:40] for d in retrieved_docs[:5]]}")
                first_sample_printed = True
            
            # Debug tracking
            if len(supporting_titles) == 0:
                queries_with_empty_supporting += 1
            if num_hard == 0:
                queries_with_no_hard += 1
            
            total_correct_in_baseline += num_correct
            total_hard_found += num_hard

            contexts = relevant_contexts + hard_contexts

            prompt = create_prompt(question, contexts)
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

            pred_text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = extract_short_answer(pred_text)

            em = exact_match_score(pred, gold)
            f1 = f1_score(pred, gold)

            em_count += em
            f1_sum += f1

            results.append({
                "query_id": qid,
                "question": question,
                "gold_answer": gold,
                "generated_answer": pred,
                "k_relevant": k_rel,
                "k_hard": k_hard,
                "num_contexts": len(contexts),
                "exact_match": em,
                "f1_score": f1,
            })

        n = len(results)
        em_score = em_count / n if n else 0.0
        f1_avg = f1_sum / n if n else 0.0

        metrics = {
            "config_name": name,
            "k_relevant": k_rel,
            "k_hard": k_hard,
            "num_queries": n,
            "skipped": skipped,
            "exact_match": em_score,
            "f1_score": f1_avg,
            "exact_matches_count": em_count,
            "retrieval_file": retrieval_path
        }
        all_summary[name] = metrics

        out_results = os.path.join(BASE_DIR, RESULTS_DIR, f"hard_negatives_{name}.json")
        out_metrics = os.path.join(BASE_DIR, RESULTS_DIR, f"hard_negatives_{name}_metrics.json")
        save_json(results, out_results)
        save_json(metrics, out_metrics)

        print(f"\nEM: {em_score:.4f} ({em_count}/{n}) | F1: {f1_avg:.4f} | skipped: {skipped}")
        print(f"\n🔍 DEBUG INFO:")
        print(f"  Queries with empty supporting_titles: {queries_with_empty_supporting}/{n}")
        print(f"  Queries with NO hard negatives found: {queries_with_no_hard}/{n}")
        print(f"  Average correct docs in baseline top-{k_rel}: {total_correct_in_baseline/n:.2f}/{k_rel}")
        print(f"  Average hard negatives added: {total_hard_found/n:.2f}/{k_hard}")
        print(f"  Baseline retrieval precision@{k_rel}: {(total_correct_in_baseline/(n*k_rel)*100):.1f}%")
        print("\nSaved:", out_results)
        print("Saved:", out_metrics)

    summary_path = os.path.join(BASE_DIR, RESULTS_DIR, "hard_negatives_summary.json")
    save_json(all_summary, summary_path)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<25} {'k_rel':<6} {'k_hard':<7} {'EM':<10} {'F1':<10}")
    print("-" * 80)
    for name, m in all_summary.items():
        print(f"{name:<25} {m['k_relevant']:<6} {m['k_hard']:<7} {m['exact_match']:<10.4f} {m['f1_score']:<10.4f}")
    print("\nSaved summary:", summary_path)

if __name__ == "__main__":
    main()

"""
Evaluate Retrieval Results - Answer Generation

Generates answers using Flan-T5 based on retrieved documents 
and evaluates with exact match and F1 scores.

Evaluates top-1, top-3, and top-5 retrieval results.
"""

import os, json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import centralized utils
from utils.evaluation import normalize_answer, exact_match_score, f1_score
from utils.prompts import create_prompt, extract_short_answer
from utils.data_loading import load_json

# -------------------------
# Config
# -------------------------
DATA_PATH   = "data/dev.json"
OUTPUT_DIR  = "output"
RESULTS_DIR = "results_new"
LIMIT       = 1200

os.makedirs(RESULTS_DIR, exist_ok=True)

# Define k values to evaluate
K_VALUES = [1, 3, 5]

def main():
    print("=" * 80)
    print("RETRIEVAL EVALUATION - Answer Generation")
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

    # load Flan-T5 locally
    FLANT5_PATH = os.path.join(BASE_DIR, "models", "flan-t5-base")
    print("\nLoading Flan-T5 locally:", FLANT5_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FLANT5_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_PATH).to(device)
    model.eval()
    print("✓ Model loaded")

    # load data
    dev_data = load_json(os.path.join(BASE_DIR, DATA_PATH))[:LIMIT]
    print(f"✓ Loaded {len(dev_data)} dev questions")

    all_summary = {}

    # Evaluate each k value
    for k in K_VALUES:
        print(f"\n{'='*80}")
        print(f"EVALUATING TOP-{k} RETRIEVAL")
        print(f"{'='*80}")

        # Load retrieval results
        retrieval_path = os.path.join(BASE_DIR, OUTPUT_DIR, f"top_{k}_retrieval_results.json")
        if not os.path.exists(retrieval_path):
            print(f"⚠ Warning: {retrieval_path} not found, skipping k={k}")
            continue
            
        retrieval_data = load_json(retrieval_path)
        retr_by_qid = {x["query_id"]: x for x in retrieval_data}
        print(f"✓ Loaded retrieval results: {len(retrieval_data)} queries")

        results = []
        em_count = 0
        f1_sum = 0.0

        for item in tqdm(dev_data, desc=f"Top-{k}"):
            qid = item["_id"]
            if qid not in retr_by_qid:
                continue

            question = item["question"]
            gold = item["answer"]

            # Get retrieved documents
            retrieved_docs = retr_by_qid[qid]["retrieved_docs"]
            contexts = [d["text"] for d in retrieved_docs[:k]]

            # Create prompt and generate answer
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

            pred = extract_short_answer(tokenizer.decode(out[0], skip_special_tokens=True))
            em = exact_match_score(pred, gold)
            f1 = f1_score(pred, gold)

            em_count += em
            f1_sum += f1

            results.append({
                "query_id": qid,
                "question": question,
                "gold_answer": gold,
                "generated_answer": pred,
                "k_relevant": k,
                "k_random": 0,
                "exact_match": em,
                "f1_score": f1,
            })

        n = len(results)
        metrics = {
            "config_name": f"top_{k}",
            "k_relevant": k,
            "num_queries": n,
            "exact_match": em_count / n if n else 0.0,
            "f1_score": f1_sum / n if n else 0.0,
            "exact_matches_count": em_count
        }

        all_summary[f"top_{k}"] = metrics

        # Save results
        answers_file = os.path.join(RESULTS_DIR, f"top_{k}_answers.json")
        metrics_file = os.path.join(RESULTS_DIR, f"top_{k}_metrics.json")
        
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"✓ EM: {metrics['exact_match']:.4f} | F1: {metrics['f1_score']:.4f}")
        print(f"✓ Saved to {answers_file}")

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "retrieval_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for k, metrics in all_summary.items():
        print(f"{k}: EM={metrics['exact_match']:.4f}, F1={metrics['f1_score']:.4f}")
    print(f"✓ Summary saved to {summary_file}")
    print("\n✓ Finished Retrieval Evaluation")

if __name__ == "__main__":
    main()

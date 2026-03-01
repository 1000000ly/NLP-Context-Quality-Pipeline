"""
Step 3: Random Negative Sampling

Combines top-k retrieved documents with randomly sampled irrelevant documents
from the corpus. Tests whether adding random noise affects QA performance (RQ1).

Uses:
- Off-the-shelf retriever outputs (Contriever)
- Fixed generator (Flan-T5, local)
"""

import os, json, random, re, string
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
CORPUS_PATH = "data/clean_wiki_corpus.json"
RESULTS_DIR = "results"
LIMIT       = 1200

CONFIGS = [
    {"k_relevant": 1, "k_random": 1, "name": "1plus1random"},
    {"k_relevant": 3, "k_random": 3, "name": "3plus3random"},
    {"k_relevant": 5, "k_random": 5, "name": "5plus5random"},
]
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("STEP 3: RANDOM NEGATIVE SAMPLING")
    print("=" * 80)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # base dir
    # NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    # BASE_DIR = NOTEBOOK_DIR if os.path.exists(os.path.join(NOTEBOOK_DIR, "data")) else os.path.dirname(NOTEBOOK_DIR)
    BASE_DIR = ""

    # load Flan-T5 locally
    FLANT5_PATH = os.path.join(BASE_DIR, "models", "flan-t5-base")
    print("\nLoading Flan-T5 locally:", FLANT5_PATH)
    tokenizer = AutoTokenizer.from_pretrained(FLANT5_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_PATH).to(device)
    model.eval()

    # load data
    dev_data = load_json(os.path.join(BASE_DIR, DATA_PATH))[:LIMIT]
    print(f"Loaded dev: {len(dev_data)}")

    # load corpus (dict -> list of texts)
    corpus_raw = load_json(os.path.join(BASE_DIR, CORPUS_PATH))
    corpus_texts = [v["text"] for v in corpus_raw.values()]
    print(f"Loaded corpus docs: {len(corpus_texts)}")

    all_summary = {}

    for cfg in CONFIGS:
        k_rel = cfg["k_relevant"]
        k_rand = cfg["k_random"]
        name = cfg["name"]

        retrieval_path = os.path.join(BASE_DIR, "output", f"top_{k_rel}_retrieval_results.json")
        if not os.path.exists(retrieval_path):
            print(f"⚠ Warning: {retrieval_path} not found, skipping")
            continue
        retrieval_data = load_json(retrieval_path)
        retr_by_qid = {x["query_id"]: x for x in retrieval_data}

        print(f"\n{'='*80}")
        print(f"CONFIG: {name} (top-{k_rel} + {k_rand} random)")
        print(f"{'='*80}")

        results = []
        em_count = 0
        f1_sum = 0.0

        for item in tqdm(dev_data, desc=name):
            qid = item["_id"]
            if qid not in retr_by_qid:
                continue

            question = item["question"]
            gold = item["answer"]

            retrieved_docs = retr_by_qid[qid]["retrieved_docs"]
            relevant_contexts = [d["text"] for d in retrieved_docs[:k_rel]]

            # random negatives (exclude retrieved)
            retrieved_set = set(relevant_contexts)
            random_pool = [c for c in corpus_texts if c not in retrieved_set]
            random_contexts = random.sample(random_pool, min(k_rand, len(random_pool)))

            contexts = relevant_contexts + random_contexts
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
                "k_relevant": k_rel,
                "k_random": k_rand,
                "exact_match": em,
                "f1_score": f1,
            })

        n = len(results)
        metrics = {
            "config_name": name,
            "k_relevant": k_rel,
            "k_random": k_rand,
            "num_queries": n,
            "exact_match": em_count / n if n else 0.0,
            "f1_score": f1_sum / n if n else 0.0,
            "exact_matches_count": em_count
        }

        all_summary[name] = metrics

        # Save results
        with open(os.path.join(RESULTS_DIR, f"random_negatives_{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(RESULTS_DIR, f"random_negatives_{name}_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"EM: {metrics['exact_match']:.4f} | F1: {metrics['f1_score']:.4f}")

    # Save summary
    with open(os.path.join(RESULTS_DIR, "random_negatives_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("\n✓ Finished Step 3 (Random Negatives)")

if __name__ == "__main__":
    main()

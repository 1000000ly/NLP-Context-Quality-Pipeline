"""
Oracle Evaluation - Upper Bound Performance

Uses ground truth supporting_facts to select the perfect documents,
then generates answers and evaluates. This gives the upper bound performance
when the retriever is perfect.

Input: data/dev.json with supporting_facts
Output: results/oracle_*.json with answers and metrics
"""

import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import centralized utils
from utils.evaluation import exact_match_score, f1_score
from utils.prompts import create_prompt, extract_short_answer
from utils.data_loading import load_json

# -------------------------
# Config
# -------------------------
DATA_PATH = "data/dev.json"
CORPUS_PATH = "data/clean_wiki_corpus.json"
RESULTS_DIR = "results"
LIMIT = 1200

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_oracle_contexts(item, corpus_dict):
    """
    Extract oracle contexts using supporting_facts.
    
    supporting_facts format: [["title1", sent_idx], ["title2", sent_idx], ...]
    Returns list of full document texts for those titles.
    """
    supporting_facts = item.get("supporting_facts", [])
    oracle_titles = set()
    
    # Extract unique titles from supporting facts
    for fact in supporting_facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 1:
            oracle_titles.add(fact[0])
    
    # Get full document texts from corpus
    oracle_contexts = []
    for title in oracle_titles:
        # Find document in corpus by title
        for doc_id, doc_info in corpus_dict.items():
            if doc_info.get("title") == title:
                oracle_contexts.append(doc_info.get("text", ""))
                break
    
    return oracle_contexts, list(oracle_titles)


def main():
    print("=" * 80)
    print("ORACLE EVALUATION - Upper Bound with Ground Truth Documents")
    print("=" * 80)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    BASE_DIR = ""

    # Load Flan-T5
    FLANT5_PATH = os.path.join(BASE_DIR, "models", "flan-t5-base")
    print(f"\nLoading Flan-T5 from: {FLANT5_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(FLANT5_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_PATH).to(device)
    model.eval()
    print("✓ Flan-T5 loaded")

    # Load data
    dev_data = load_json(os.path.join(BASE_DIR, DATA_PATH))[:LIMIT]
    print(f"✓ Loaded {len(dev_data)} dev questions")

    # Load corpus
    corpus_dict = load_json(os.path.join(BASE_DIR, CORPUS_PATH))
    print(f"✓ Loaded corpus: {len(corpus_dict)} documents")
    
    # Create title index for faster lookup
    print("Creating title index...")
    title_to_doc = {}
    for doc_id, doc_info in corpus_dict.items():
        title = doc_info.get("title")
        if title:
            title_to_doc[title] = doc_info.get("text", "")
    print(f"✓ Indexed {len(title_to_doc)} titles")

    print(f"\n{'='*80}")
    print("PROCESSING ORACLE PREDICTIONS")
    print(f"{'='*80}")

    results = []
    em_count = 0
    f1_sum = 0.0
    queries_with_no_oracle = 0
    queries_with_missing_docs = 0
    total_oracle_docs = 0

    for item in tqdm(dev_data, desc="Oracle Evaluation"):
        qid = item["_id"]
        question = item["question"]
        gold = item["answer"]
        
        # Get supporting facts
        supporting_facts = item.get("supporting_facts", [])
        oracle_titles = set()
        
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and len(fact) >= 1:
                oracle_titles.add(fact[0])
        
        if not oracle_titles:
            queries_with_no_oracle += 1
            # Generate with empty context
            contexts = []
        else:
            # Get documents from corpus
            contexts = []
            missing_titles = []
            
            for title in oracle_titles:
                if title in title_to_doc:
                    contexts.append(title_to_doc[title])
                else:
                    missing_titles.append(title)
            
            if missing_titles:
                queries_with_missing_docs += 1
            
            total_oracle_docs += len(contexts)
        
        # Generate answer
        if contexts:
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
        else:
            pred = ""
        
        # Evaluate
        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)
        
        em_count += em
        f1_sum += f1
        
        results.append({
            "query_id": qid,
            "question": question,
            "gold_answer": gold,
            "generated_answer": pred,
            "oracle_titles": list(oracle_titles),
            "num_oracle_docs": len(contexts),
            "exact_match": em,
            "f1_score": f1,
        })

    # Calculate metrics
    n = len(results)
    em_score = em_count / n if n else 0.0
    f1_avg = f1_sum / n if n else 0.0
    avg_oracle_docs = total_oracle_docs / n if n else 0.0

    metrics = {
        "config_name": "oracle",
        "num_queries": n,
        "exact_match": em_score,
        "f1_score": f1_avg,
        "exact_matches_count": em_count,
        "avg_oracle_docs_per_query": avg_oracle_docs,
        "queries_with_no_oracle_docs": queries_with_no_oracle,
        "queries_with_missing_docs": queries_with_missing_docs,
    }

    # Save results
    answers_file = os.path.join(RESULTS_DIR, "oracle_answers.json")
    metrics_file = os.path.join(RESULTS_DIR, "oracle_metrics.json")
    
    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("ORACLE EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total queries: {n}")
    print(f"Exact Match: {em_score:.4f} ({em_count}/{n})")
    print(f"F1 Score: {f1_avg:.4f}")
    print(f"Average oracle docs per query: {avg_oracle_docs:.2f}")
    print(f"Queries with no oracle docs: {queries_with_no_oracle}")
    print(f"Queries with missing docs in corpus: {queries_with_missing_docs}")
    print(f"\n✓ Saved answers to: {answers_file}")
    print(f"✓ Saved metrics to: {metrics_file}")
    print("="*80)


if __name__ == "__main__":
    main()

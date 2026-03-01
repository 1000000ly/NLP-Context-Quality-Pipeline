"""
Evaluate Oracle predictions with proper EM and F1 metrics.
"""
import json
from utils.evaluation import normalize_answer, exact_match_score, f1_score
from utils.data_loading import load_json

# Configuration
ORACLE_PREDICTIONS = "output/oracle_predictions_flant5.json"
OUTPUT_METRICS = "results/oracle_metrics.json"
OUTPUT_RESULTS = "results/oracle_results_with_metrics.json"

def main():
    print("="*80)
    print("EVALUATING ORACLE PREDICTIONS")
    print("="*80)
    
    # Load oracle predictions
    print(f"\nLoading predictions from: {ORACLE_PREDICTIONS}")
    predictions = load_json(ORACLE_PREDICTIONS)
    print(f"✓ Loaded {len(predictions)} predictions")
    
    # Calculate metrics for each prediction
    results = []
    exact_matches = 0
    total_f1 = 0.0
    empty_count = 0
    total_gen_length = 0
    total_gold_length = 0
    
    for pred in predictions:
        generated = pred.get('generated_answer', '')
        gold = pred.get('gold_answer', '')
        
        # Skip error cases
        if generated == "ERROR" or 'error' in pred:
            continue
        
        # Compute metrics
        em = exact_match_score(generated, gold)
        f1 = f1_score(generated, gold)
        
        if em:
            exact_matches += 1
        total_f1 += f1
        
        if not generated.strip():
            empty_count += 1
        
        total_gen_length += len(generated.split())
        total_gold_length += len(gold.split())
        
        results.append({
            "_id": pred['_id'],
            "question": pred['question'],
            "generated_answer": generated,
            "gold_answer": gold,
            "exact_match": em,
            "f1_score": f1,
            "context_used": "oracle"
        })
    
    # Calculate aggregate metrics
    num_queries = len(results)
    em_score = exact_matches / num_queries if num_queries > 0 else 0
    avg_f1 = total_f1 / num_queries if num_queries > 0 else 0
    avg_gen_length = total_gen_length / num_queries if num_queries > 0 else 0
    avg_gold_length = total_gold_length / num_queries if num_queries > 0 else 0
    empty_pct = (empty_count / num_queries * 100) if num_queries > 0 else 0
    
    metrics = {
        "model": "oracle",
        "context": "supporting_facts_only",
        "num_queries": num_queries,
        "exact_match": em_score,
        "f1_score": avg_f1,
        "exact_matches_count": exact_matches,
        "empty_answers_count": empty_count,
        "empty_answers_pct": empty_pct,
        "avg_generated_length": avg_gen_length,
        "avg_gold_length": avg_gold_length
    }
    
    # Save results with metrics
    import os
    os.makedirs("results", exist_ok=True)
    
    with open(OUTPUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_METRICS, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print(f"\n{'='*80}")
    print("ORACLE EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total queries: {num_queries}")
    print(f"\nPerformance:")
    print(f"  Exact Match (EM): {em_score:.4f} ({exact_matches}/{num_queries})")
    print(f"  F1 Score:         {avg_f1:.4f}")
    print(f"\nAnswer Statistics:")
    print(f"  Empty answers:    {empty_count} ({empty_pct:.2f}%)")
    print(f"  Avg gen length:   {avg_gen_length:.2f} words")
    print(f"  Avg gold length:  {avg_gold_length:.2f} words")
    print(f"\nOutput Files:")
    print(f"  Metrics:  {OUTPUT_METRICS}")
    print(f"  Results:  {OUTPUT_RESULTS}")
    print(f"{'='*80}")
    
    return metrics

if __name__ == "__main__":
    main()

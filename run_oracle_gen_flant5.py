import json
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import centralized utils
from utils.evaluation import normalize_answer, exact_match_score, f1_score
from utils.prompts import create_prompt, extract_short_answer
from utils.data_loading import load_json as utils_load_json

# ================= Configuration =================
DATA_PATH = "data/dev.json"
OUTPUT_PATH = "output/oracle_predictions_flant5.json"
MODEL_NAME = "google/flan-t5-base"
LIMIT = 1200 
# ===============================================

def load_data(filepath, limit=None):
    print(f"Loading data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        print(f"✓ Loaded {len(data)} items")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        exit(1)

def create_prompt(question: str, context: str) -> str:
    """Create concise prompt for answer generation."""
    return f"""Answer the question in the shortest form possible (a few words). Use only the context provided.

Context: {context}

Question: {question}

Answer:"""

def main():
    print("="*80)
    print("ORACLE ANSWER GENERATION (Using Gold Contexts)")
    print("="*80)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\nUsing device: {device}")
    print(f"Model: {MODEL_NAME}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load data
    data = load_data(DATA_PATH, limit=LIMIT)
    
    results = []
    print(f"\n{'='*80}")
    print("Generating answers...")
    print(f"{'='*80}\n")
    
    for item in tqdm(data, desc="Oracle generation"):
        question = item['question']
        
        # Extract ONLY supporting facts (oracle = gold standard contexts)
        context_text = ""
        if 'supporting_facts' in item and 'context' in item:
            # Build a dict for quick lookup
            context_dict = {doc[0]: doc[1] for doc in item['context'] if len(doc) >= 2}
            
            # Extract only the supporting fact sentences
            for fact in item['supporting_facts']:
                doc_title = fact[0]
                sent_idx = fact[1]
                
                if doc_title in context_dict and sent_idx < len(context_dict[doc_title]):
                    sentence = context_dict[doc_title][sent_idx]
                    context_text += f"{sentence} "
        
        context_text = context_text.strip()
        
        # Create prompt
        prompt = create_prompt(question, context_text)
        
        # Generate answer
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            results.append({
                "_id": item['_id'],
                "question": question,
                "gold_answer": item['answer'],
                "generated_answer": generated_answer,
                "context_used": "oracle"
            })
            
        except Exception as e:
            print(f"\nError processing {item.get('_id', 'unknown')}: {e}")
            results.append({
                "_id": item['_id'],
                "question": question,
                "gold_answer": item['answer'],
                "generated_answer": "ERROR",
                "error": str(e)
            })
    
    # Save results
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Calculate basic metrics
    correct = 0
    for result in results:
        pred = result['generated_answer'].lower().strip()
        gold = result['gold_answer'].lower().strip()
        if pred in gold or gold in pred:
            correct += 1
    
    accuracy = correct / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print("ORACLE GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total queries: {len(results)}")
    print(f"Approximate matches: {correct}/{len(results)} ({accuracy:.2%})")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

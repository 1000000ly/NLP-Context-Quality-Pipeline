import sys
import os
import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation

# Set environment variable to help with CUDA memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "DEXTER")

# Adjust batch size based on available memory
BATCH_SIZE = 256 if torch.cuda.is_available() else 4

if package_path not in sys.path:
    sys.path.append(package_path)

print(f"Added {package_path} to sys.path")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")


try:
    from dexter.retriever.dense.HfRetriever import HfRetriever
    from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
    from dexter.data.datastructures.evidence import Evidence
    from dexter.data.datastructures.question import Question
    from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
    print("✓ Successfully imported DEXTER modules")
except ImportError as e:
    print(f"✗ Error importing DEXTER modules: {e}")
    print(f"  Make sure you've installed the package: pip install -e {package_path}")
    sys.exit(1)

from utils.data_loading import load_json as load_json_data

# def load_json_data(file_path):
#     """Load JSON data from file with error handling."""
#     print(f"\nLoading data from: {file_path}...")
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         print(f"✓ Loaded {len(data)} items\n")
#         return data
#     except FileNotFoundError:
#         print(f"✗ Error: File not found - {file_path}")
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"✗ Error: Invalid JSON format - {e}")
#         sys.exit(1)

def main():
    print("="*80)
    print("CONTRIEVER RETRIEVAL - Top-K Document Retrieval")
    print("="*80)
    LIMIT = 1200

    data_dir = "data"
    output_dir = "output"
    embeddings_dir = "indices/corpus"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Embeddings directory: {embeddings_dir}")
    
    dev_file = os.path.join(data_dir, "dev.json")
    corpus_file = os.path.join(data_dir, "clean_wiki_corpus.json")

    # Load corpus
    raw_corpus = load_json_data(corpus_file)

    # Convert to Evidence objects - this is what contrieve needs 
    print("Converting corpus to Evidence objects...")
    corpus = []
    for doc_id, doc_info in raw_corpus.items():
        corpus.append(Evidence(
            idx=doc_id,
            text=doc_info.get('text', ''),
            title=doc_info.get('title', '')
        ))
    print(f"✓ Created {len(corpus)} Evidence objects")

    # Load queries
    dev_data = load_json_data(dev_file)[:LIMIT]

    # again this is what Contrivever needs 
    print(f"Converting queries to Question objects...")
    queries = []
    for item in dev_data:
        queries.append(Question(
            idx=item['_id'],
            text=item['question']
        ))
    print(f"✓ Created {len(queries)} Question objects")

    # Initialize retriever
    print("\n" + "="*80)
    print("Initializing Contriever Retriever...")
    print("="*80)
    # config = DenseHyperParams(
    #     query_encoder_path="models/contriever",
    #     document_encoder_path="models/contriever",
    #     batch_size=BATCH_SIZE,
    #     show_progress_bar=True
    # )
    
    # try:
    #     retriever = HfRetriever(config)
    #     print("✓ Retriever initialized successfully")
    # except Exception as e:
    #     print(f"✗ Error initializing retriever: {e}")
    #     sys.exit(1)

    CONTRIEVER_PATH = os.path.join("", "models/contriever")
    if os.path.exists(CONTRIEVER_PATH):
        print("\nLoading Contriever from local:", CONTRIEVER_PATH)
        doc_encoder = SentenceTransformer(CONTRIEVER_PATH)
        query_encoder = SentenceTransformer(CONTRIEVER_PATH)
    else:
        print("\nLoading Contriever from HF: facebook/contriever (requires internet)")
        doc_encoder = SentenceTransformer("facebook/contriever")
        query_encoder = SentenceTransformer("facebook/contriever")

    # Run retrieval
    print("\n" + "="*80)
    print("Starting Retrieval Process...")
    print("="*80)
    score_function = CosineSimilarity()
    
    # Retrieve more documents to have enough for all experiments
    # (positives + hard negatives + random negatives)
    max_k = 20
    k_values = [1, 3, 5, 10, 20]
    
    print(f"Retrieving top-{max_k} documents for {len(queries)} queries...")
    try:
        # results = retriever.retrieve(corpus, queries, max_k, score_function)
        # Prepare corpus data
        corpus_texts = [doc.text() for doc in corpus]
        corpus_ids = [doc.id() for doc in corpus]
        
        # Check if corpus embeddings already exist
        embeddings_file = os.path.join(embeddings_dir, "embeddings.pt")
        corpus_ids_file = os.path.join(embeddings_dir, "corpus_ids.json")
        
        if os.path.exists(embeddings_file) and os.path.exists(corpus_ids_file):
            print(f"\nLoading existing corpus embeddings from {embeddings_file}...")
            doc_embeddings = torch.load(embeddings_file)
            with open(corpus_ids_file, 'r') as f:
                saved_corpus_ids = json.load(f)
            
            # Verify the corpus hasn't changed
            if saved_corpus_ids == corpus_ids:
                print(f"✓ Loaded {doc_embeddings.shape[0]} corpus embeddings")
            else:
                print("⚠ Corpus has changed, recomputing embeddings...")
                doc_embeddings = doc_encoder.encode(
                    corpus_texts,
                    batch_size=BATCH_SIZE,
                    convert_to_tensor=True,
                    show_progress_bar=True
                )
                # Save the new embeddings
                print(f"Saving corpus embeddings to {embeddings_file}...")
                torch.save(doc_embeddings, embeddings_file)
                with open(corpus_ids_file, 'w') as f:
                    json.dump(corpus_ids, f)
                print("✓ Embeddings saved")
        else:
            print("\nComputing corpus embeddings (first time)...")
            doc_embeddings = doc_encoder.encode(
                corpus_texts,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            # Save embeddings for future use
            print(f"Saving corpus embeddings to {embeddings_file}...")
            torch.save(doc_embeddings, embeddings_file)
            with open(corpus_ids_file, 'w') as f:
                json.dump(corpus_ids, f)
            print("✓ Embeddings saved")
        
        # Encode queries (always computed fresh)
        print("\nEncoding queries...")
        query_texts = [q.text() for q in queries]
        query_ids = [q.id() for q in queries]
        
        query_embeddings = query_encoder.encode(
            query_texts,
            batch_size=BATCH_SIZE,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Compute cosine similarity
        scores = torch.matmul(
            torch.nn.functional.normalize(query_embeddings, dim=1),
            torch.nn.functional.normalize(doc_embeddings, dim=1).T
        )
        
        # Build results dict (DEXTER-compatible)
        results = {}
        for qi, qid in enumerate(query_ids):
            top_scores, top_idx = torch.topk(scores[qi], max_k)
            results[qid] = {
                corpus_ids[di]: float(top_scores[i])
                for i, di in enumerate(top_idx.tolist())
            }

        print(f"✓ Retrieval completed successfully")
        
    except Exception as e:
        print(f"✗ Error during retrieval: {e}")
        sys.exit(1)
    
    # Create corpus lookup dictionary for fast access
    print("\n" + "="*80)
    print("Processing and Saving Results...")
    print("="*80)
    print("Creating corpus lookup dictionary...")
    corpus_dict = {doc.id(): doc for doc in corpus}
    
    print("Formatting retrieval results...")
    all_query_results = []
    queries_without_results = 0
    
    for query in queries:
        q_id = query.id()
        query_result = {
            "query_id": q_id,
            "query_text": query.text(),
            "retrieved_docs": []
        }
        
        if q_id in results:
            # Sort by score (descending)
            sorted_docs = sorted(results[q_id].items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs[:max_k], 1):
                doc = corpus_dict.get(doc_id)
                if doc:
                    query_result["retrieved_docs"].append({
                        "rank": rank,
                        "doc_id": doc_id,
                        "score": float(score),
                        "title": doc.title() if doc.title() else "No title",
                        "text": doc.text()
                    })
        else:
            queries_without_results += 1
        
        all_query_results.append(query_result)
    
    if queries_without_results > 0:
        print(f"⚠ Warning: {queries_without_results} queries had no results")
    
    # Save results for each k value
    print(f"\nSaving results for k = {k_values}...")
    for k in k_values:
        k_results = []
        for query_result in all_query_results:
            k_results.append({
                "query_id": query_result["query_id"],
                "query_text": query_result["query_text"],
                "retrieved_docs": query_result["retrieved_docs"][:k]
            })
        
        output_file = os.path.join(output_dir, f"top_{k}_retrieval_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(k_results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved top-{k} results to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("RETRIEVAL SUMMARY")
    print("="*80)
    print(f"Total queries processed: {len(queries)}")
    print(f"Total documents in corpus: {len(corpus)}")
    print(f"K values: {k_values}")
    print(f"Results saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()

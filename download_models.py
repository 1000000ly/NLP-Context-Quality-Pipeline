import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = "/home/asasin/NLP_Project"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CONTRIEVER_PATH = os.path.join(MODELS_DIR, "contriever")
FLANT5_PATH = os.path.join(MODELS_DIR, "flan-t5-base")

print("Downloading model weights...")
print("="*80)

# Download Contriever
if not os.path.exists(CONTRIEVER_PATH):
    print("\nDownloading facebook/contriever...")
    model = SentenceTransformer('facebook/contriever')
    model.save(CONTRIEVER_PATH)
    print(f"✓ Contriever saved to: {CONTRIEVER_PATH}")
else:
    print(f"✓ Contriever already exists at: {CONTRIEVER_PATH}")

# Download Flan-T5
if not os.path.exists(FLANT5_PATH):
    print("\nDownloading google/flan-t5-base...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    tokenizer.save_pretrained(FLANT5_PATH)
    model.save_pretrained(FLANT5_PATH)
    print(f"✓ Flan-T5 saved to: {FLANT5_PATH}")
else:
    print(f"✓ Flan-T5 already exists at: {FLANT5_PATH}")

print("="*80)
print("✓ All models downloaded and cached locally")

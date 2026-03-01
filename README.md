# Impact of Context Quality on Retrieval-Augmented Generation for Complex QA

## Overview
This repository contains a diagnostic study on the robustness of Retrieval-Augmented Generation (RAG) systems when processing complex, multi-hop questions. Using the 2WikiMultiHopQA dataset, the pipeline evaluates how different noise types—specifically random negatives and semantically related hard negatives—influence the accuracy of downstream answer generation.

## Technical Implementation
* **Data Scale**: Processes a Wikipedia-based corpus consisting of 526,329 documents.
* **Retrieval Optimization**: Utilizes Contriever as a dense retriever, with query encoder fine-tuning performed via the ADORE framework to improve ranking precision.
* **Generation Model**: Employs Flan-T5-Base as a fixed generator conditioned on retrieved contexts.
* **Infrastructure**: Experiments conducted using NVIDIA RTX 4090 GPU hardware on a cloud computing platform.
* **Environment Management**: Project dependencies and reproducibility are managed via `uv.lock` and `pyproject.toml`.

## Key Findings
* **Performance Gap**: Gold-only oracle contexts achieved 34.92% Exact Match (EM) accuracy, compared to 21.75% for the best mixed-context retrieval setting, revealing a 13.17-point performance gap.
* **Noise Sensitivity**: Random noise is significantly more detrimental to generation than semantically related hard negatives. At a retrieval depth of k=5, random noise caused a 4.25% drop in EM, whereas hard negatives resulted in a negligible 0.42% decrease.
* **Semantic Coherence**: Results indicate that generators effectively tolerate factual distractions as long as the context remains topically consistent, but struggle with the semantic discontinuity introduced by random noise.


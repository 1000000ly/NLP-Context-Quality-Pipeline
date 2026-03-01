# Impact of Context Quality on Retrieval-Augmented Generation for Complex QA

## Overview
This repository contains a diagnostic study on the robustness of Retrieval-Augmented Generation (RAG) systems when processing complex, multi-hop questions. [cite_start]Using the 2WikiMultiHopQA dataset, the pipeline evaluates how different noise types—specifically random negatives and semantically related hard negatives—influence the accuracy of downstream answer generation[cite: 1, 31, 34].

## Technical Implementation
* [cite_start]**Data Scale**: Processes a Wikipedia-based corpus consisting of 526,329 documents[cite: 93].
* [cite_start]**Retrieval Optimization**: Utilizes Contriever as a dense retriever, with query encoder fine-tuning performed via the ADORE framework to improve ranking precision[cite: 56, 62, 104].
* [cite_start]**Generation Model**: Employs Flan-T5-Base as a fixed generator conditioned on retrieved contexts[cite: 77].
* [cite_start]**Infrastructure**: Experiments conducted using NVIDIA RTX 4090 GPU hardware on a cloud computing platform[cite: 115].
* **Environment Management**: Project dependencies and reproducibility are managed via `uv.lock` and `pyproject.toml`.

## Key Findings
* [cite_start]**Performance Gap**: Gold-only oracle contexts achieved 34.92% Exact Match (EM) accuracy, compared to 21.75% for the best mixed-context retrieval setting, revealing a 13.17-point performance gap[cite: 10].
* **Noise Sensitivity**: Random noise is significantly more detrimental to generation than semantically related hard negatives. [cite_start]At a retrieval depth of k=5, random noise caused a 4.25% drop in EM, whereas hard negatives resulted in a negligible 0.42% decrease[cite: 125, 130].
* [cite_start]**Semantic Coherence**: Results indicate that generators effectively tolerate factual distractions as long as the context remains topically consistent, but struggle with the semantic discontinuity introduced by random noise[cite: 138, 158].

## Project Structure
* `NLP_2025_project 1.pdf`: Full technical report detailing methodology and quantitative analysis.
* `clean_corpus.py`: Script for cleaning and normalizing the primary data corpus.
* `evaluate_oracle.py`: Establishing the performance upper bound using annotated gold documents.
* `step6_adore_training.ipynb`: Implementation of the ADORE fine-tuning process for optimized retrieval.
* `run_answer_generation.py`: Core script for executing the end-to-end RAG inference pipeline.

## Setup
This project uses `uv` for Python package management. To install dependencies and synchronize the environment:
```bash
uv sync

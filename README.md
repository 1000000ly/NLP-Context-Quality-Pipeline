Impact of Context Quality on Retrieval-Augmented Generation for Complex QA
Overview
This project presents a diagnostic study on the robustness of Retrieval-Augmented Generation (RAG) systems when faced with varying levels of context quality. Using the 2WikiMultiHopQA dataset, the pipeline evaluates how different noise types—specifically random negatives and semantically related hard negatives—influence the accuracy of multi-hop answer generation. The study identifies a significant performance gap between standard retrieval-based systems and an oracle upper bound.
+4

Technical Implementation

Data Scale: The system processes a Wikipedia-based corpus consisting of 526,329 documents.


Data Preprocessing: Document titles are normalized and deduplicated to ensure data integrity.


Retrieval and Optimization: The pipeline utilizes Contriever as a dense retriever, with query encoder fine-tuning performed via the ADORE framework to improve ranking precision.
+2


Generation: Flan-T5-Base serves as the fixed generator, conditioned on retrieved contexts to produce concise answers.
+1


Infrastructure: Experiments were conducted using NVIDIA RTX 4090 GPU hardware on a cloud computing platform.

Environment Management: Dependency management and environment reproducibility are handled via uv.lock and pyproject.toml.

Key Findings

The Oracle Gap: Gold-only oracle contexts achieved 34.92% Exact Match (EM) accuracy, compared to 21.75% for the best mixed-context retrieval setting, revealing a 13.17-point performance ceiling.


Noise Sensitivity: Random noise is significantly more detrimental to generation than semantically related hard negatives. At a retrieval depth of k=5, random noise caused a 4.25% drop in accuracy, while hard negatives resulted in a negligible 0.42% decrease.
+3


Semantic Coherence: The findings suggest that as long as the retrieved context remains topically consistent, the generator can effectively tolerate factual distractions. However, it struggles with the semantic discontinuity introduced by unrelated random noise.

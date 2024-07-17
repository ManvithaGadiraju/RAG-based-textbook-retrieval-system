# RAG-based-textbook-retrieval-system

This repository contains code for a Textbook Retrieval System that utilizes content extraction, hierarchical tree-based indexing, retrieval techniques, multi-document/topic/section-based RAG (Retrieval Augmented Generation), and question answering components.

## Features

- **Content Extraction:** Extracts content from a hierarchical textbook structure defined in `textbook_structure.json`.
- **Indexing and Retrieval:** Uses BM25, BERT-based sentence embeddings, and DPR (Dense Passage Retrieval) for efficient document retrieval.
- **Multi-document RAG:** Retrieves relevant sections from textbooks based on user queries.
- **Question Answering:** Provides answers to user questions based on retrieved content.

## Setup Instructions

### Dependencies

- Python 3.x
- Libraries: `rank_bm25`, `sentence_transformers`, `transformers`, `torch`

Install dependencies using pip:

```bash
pip install -r requirements.txt

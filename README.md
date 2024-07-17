# RAG-based-textbook-retrieval-system

This repository contains code for a Textbook Retrieval System that utilizes content extraction, hierarchical tree-based indexing, retrieval techniques, multi-document/topic/section-based RAG (Retrieval Augmented Generation), and question answering components.

## Features

- **Content Extraction:** Extracts content from a hierarchical textbook structure defined in `textbook_structure.json`.
- **Indexing and Retrieval:** Uses BM25, BERT-based sentence embeddings, and DPR (Dense Passage Retrieval) for efficient document retrieval.
- **Multi-document RAG:** Retrieves relevant sections from textbooks based on user queries.
- **Question Answering:** Provides answers to user questions based on retrieved content.

# Selected Textbooks
- Textbook-IanGoodfellow_Yoshua Bengio_Aaron Courville ([Link to Goodfellow et al.](https://drive.google.com/file/d/1dkqroJqhtpBIJFlXJWrGkY9IzJJ31OHK/view?usp=drive_link))
- Textbook-TomMitchell ([Link to Tom Mitchell](https://drive.google.com/file/d/1wV3FvSDRc4ub29BGXjYbZYK-n3_F0oJv/view?usp=drive_link))
- Textbook-StephenMarsland ([Link to Stephen Marsland](https://drive.google.com/file/d/1bJESyreI17dyo6dszg_45Vbaax0Z34tO/view?usp=drive_link))


## Setup Instructions

### Dependencies

- Python 3.x
- Libraries: `rank_bm25`, `sentence_transformers`, `transformers`, `torch`

## Accessing the User Interface

### Prerequisites
Make sure you have Python and Streamlit installed on your system.
use command streamlit run app.py
Install dependencies using pip:

```bash
pip install -r requirements.txt


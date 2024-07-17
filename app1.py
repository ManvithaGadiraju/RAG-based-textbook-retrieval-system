import streamlit as st
import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Download necessary NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

@st.cache_resource
def load_models():
    # Initialize BERT model for embedding
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    # Initialize GPT-2 model for text generation
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    return model, gpt2_model, gpt2_tokenizer

model, gpt2_model, gpt2_tokenizer = load_models()

# Load the hierarchical tree structure
@st.cache_data
def load_tree_structure():
    with open('textbook_structure.json', 'r', encoding='utf-8') as file:
        return json.load(file)

tree_structure = load_tree_structure()

def extract_content(node):
    """Recursively extract content from the tree structure."""
    content = node['content'] + ' ' + node['summary']
    for child in node['children']:
        content += ' ' + extract_content(child)
    return content

# Extract all content from the tree
all_content = extract_content(tree_structure)

# Tokenize the content
tokenized_content = nltk.word_tokenize(all_content.lower())

# Create BM25 index
bm25 = BM25Okapi([tokenized_content])

# Create BERT embeddings for the content
content_embedding = model.encode([all_content])[0]

def expand_query(query):
    """Expand the query using synonyms and stemming."""
    expanded_query = []
    stemmer = PorterStemmer()
    
    for word in nltk.word_tokenize(query):
        expanded_query.append(word)
        # Add stemmed version
        expanded_query.append(stemmer.stem(word))
        # Add synonyms
        synsets = wordnet.synsets(word)
        for syn in synsets:
            for lemma in syn.lemmas():
                expanded_query.append(lemma.name())
    
    return ' '.join(set(expanded_query))

def hybrid_retrieval(query, k=10):
    """Perform hybrid retrieval using BM25 and BERT."""
    # Expand the query
    expanded_query = expand_query(query)
    
    # BM25 retrieval
    bm25_scores = bm25.get_scores(nltk.word_tokenize(expanded_query.lower()))
    
    # BERT retrieval
    query_embedding = model.encode([expanded_query])[0]
    bert_similarity = cosine_similarity([query_embedding], [content_embedding])[0][0]
    
    # Combine scores (you may want to adjust the weights)
    combined_score = 0.5 * bm25_scores[0] + 0.5 * bert_similarity
    
    return combined_score

def traverse_tree(node, query_embedding, results):
    """Traverse the tree and collect relevant sections."""
    node_content = node['content'] + ' ' + node['summary']
    node_embedding = model.encode([node_content])[0]
    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
    
    if similarity > 0.5:  # Adjust this threshold as needed
        results.append((node, similarity))
    
    for child in node['children']:
        traverse_tree(child, query_embedding, results)

def retrieve_relevant_sections(query, tree):
    """Retrieve relevant sections from the hierarchical tree."""
    query_embedding = model.encode([query])[0]
    results = []
    traverse_tree(tree, query_embedding, results)
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:5]]  # Return top 5 most relevant sections

def generate_response(query, context):
    """Generate a response using GPT-2 based on the query and context."""
    prompt = f"Based on the following information, answer the question.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    pad_token_id = gpt2_tokenizer.eos_token_id

    output = gpt2_model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    response = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[1].strip()

def rag_system(query):
    """Implement the RAG system."""
    # Identify relevant documents (in this case, we only have one document)
    relevance_score = hybrid_retrieval(query)
    
    # Retrieve relevant sections
    relevant_sections = retrieve_relevant_sections(query, tree_structure)
    
    # Prepare context for generation
    context = " ".join([f"{section['title']}: {section['content']}" for section in relevant_sections])
    
    # Generate response
    response = generate_response(query, context)
    
    return response, relevant_sections

# Streamlit UI
st.title("RAG-based Textbook Q&A System")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Generating answer..."):
            response, relevant_sections = rag_system(query)
        
        st.subheader("Answer:")
        st.write(response)
        
        st.subheader("Relevant Sections:")
        for section in relevant_sections:
            with st.expander(f"{section['title']}"):
                st.write(section['content'][:500] + "...")  # Display first 500 characters
    else:
        st.warning("Please enter a question.")
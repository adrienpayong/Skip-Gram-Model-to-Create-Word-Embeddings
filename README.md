# Business Objective 
The most difficult problem in the NLP (Natural Language Processing) arena is extracting context from text input, and word embeddings provide a solution by representing words as semantically relevant dense vectors.
They solve many of the issues that other approaches, such as one-hot encodings and TFIDF, have.

Even with less data, embeddings improve generalization and performance for downstream NLP applications.
So, word embedding is a feature learning approach in which words or phrases from the lexicon are mapped to real-number vectors that capture the contextual hierarchy.

General word embeddings may not function effectively across all disciplines.
As a result, in order to get better results, we must create domain-specific embeddings.
In this project, we will use Python to construct medical word embeddings using Word2vec and FastText.

Word2vec is a model that combines several models to represent dispersed representations of words in a corpus.
Word2Vec (W2V) is an algorithm that takes in a text corpus and generates a vector representation for each word.
FastText is a library developed by Facebook's Research Team to aid in the quick learning of word representations and sentence categorization.

The goal of this project is to create a search engine and Streamlit UI using the trained models (Word2Vec and FastText). 
## Data Explanation
For our research, we are evaluating a clinical trials dataset based on Covid-19.
This dataset's URL is as follows:

Link:https://dimensions.figshare.com/articles/dataset/Dimensions COVID-19 publications datasets and clinical trials/11961063
The dataset consists of 10666 rows and 21 columns.
The next two columns are critical to our success.
- Title 
- Abstract 



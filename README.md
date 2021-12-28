# Business Objective 
The most difficult problem in the NLP (Natural Language Processing) arena is extracting context from text input, and word embeddings provide a solution by representing words as semantically relevant dense vectors.
They solve many of the issues that other approaches, such as one-hot encodings and TFIDF, have. Even with less data, embeddings improve generalization and performance for downstream NLP applications.
So, word embedding is a feature learning approach in which words or phrases from the lexicon are mapped to real-number vectors that capture the contextual hierarchy.
General word embeddings may not function effectively across all disciplines. As a result, in order to get better results, we must create domain-specific embeddings.
In this project, we will use Python to construct medical word embeddings using Word2vec and FastText.

Word2vec is a model that combines several models to represent dispersed representations of words in a corpus.
Word2Vec (W2V) is an algorithm that takes in a text corpus and generates a vector representation for each word.
FastText is a library developed by Facebook's Research Team to aid in the quick learning of word representations and sentence categorization. **The goal of this project is to create a search engine with Streamlit UI using the trained models (Word2Vec and FastText).** 
## Data Explanation
For our research, we are evaluating a clinical trials dataset based on Covid-19.
This dataset's URL is as follows:

Link:https://dimensions.figshare.com/articles/dataset/Dimensions COVID-19 publications datasets and clinical trials/11961063
The dataset consists of 10666 rows and 21 columns.
The next two columns are critical to our success.
- Title 
- Abstract 
### Aim
**The project's goal is to train the Skip-gram and FastText models to do word embeddings before developing a search engine with a Streamlit UI.** 
### Tech stack

     - Language - Python
    - Libraries and Packages - pandas, numpy, matplotlib, plotly, gensim, streamlit, nltk.
    
 ### Approach 
 
  **Importing the required libraries**
  
  ```
  import streamlit as st  #importing streamlit liabrary
  import pandas as pd
  import numpy as np
  import gensim
  from gensim.models import Word2Vec
  from gensim.models import FastText
  from sklearn.decomposition import PCA
  from matplotlib import pyplot
  import matplotlib.pyplot as plt # our main display package
import plotly.graph_objects as go
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
  ```
    
   - Reading the dataset
   - Pre-processing
   - Remove URLs
   - Convert text to lower case
   - Remove numerical values
   - Remove punctuation.
   - Perform tokenization
   - Remove stop words
   - Perform lemmatization
   - Remove ‘\n’ character from the columns
   - Exploratory Data Analysis (EDA) 
   - Data Visualization using word cloud
   - Training the ‘Skip-gram’ model
   - Training the ‘FastText’ model
   - Model embeddings – Similarity
   - PCA plots for Skip-gram and FastText models
   - Convert abstract and title to vectors using the Skip-gram and FastText model
   - Use the Cosine similarity function
   - Perform input query pre-processing
   - Define a function to return top ‘n’ similar results  
   - Result evaluation
   - Run the Streamlit Application


 You can follow ![here](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Medical%20Embeddings_Final.ipynb)


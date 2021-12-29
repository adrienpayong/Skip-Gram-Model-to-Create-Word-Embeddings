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
    
   **Reading the dataset**
   ```
   df=pd.read_csv('Dimension-covid.csv')   #for preprocessing
   df1=pd.read_csv('Dimension-covid.csv')  #for returning results
   ```
   
 
   **Pre-processing**
   
   Word concepts are incomprehensible to computers.
A system for representing text is essential for the computer to recognize and interpret natural language. Word vectors are the typical technique for text representation, in which words or sentences from a specific language vocabulary are mapped to real-number vectors.
In other words, the text data is translated into a meaningful numerical representation for computers to analyze and comprehend. Certain preparation processes must be completed before vectorized text input is supplied to a machine learning/deep learning system.
The following are some general actions that are taken as needed:

- Punctuation & Stop Word removal
- Stemming
- Lemmatization
- Tokenisation

We will perform those actions 
   
 ```
   import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# function to remove all urls
def remove_urls(text):    
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

# make all text lowercase
def text_lowercase(text):
    return text.lower()

# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# lemmatize Words 
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

#Creating one function so that all functions can be applied at once
def preprocessing(text):
    
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text
 ```
 **Remove ‘\n’ character from the columns**
 ```
 #Applying preprocessing and removing '\n' character

for i in range(df.shape[0]):
    df['Abstract'][i]=preprocessing(str(df['Abstract'][i])) 
for text in df.Abstract:

 text=text.replace('\n',' ') 
  ```
 
    
   
 **Exploratory Data Analysis (EDA)**
 
 ![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Capture1.PNG)
 
**Data Visualization using word cloud**
 
 Word Clouds or Tag Clouds are text visualization techniques that are often used to visualize tags or phrases from webpages.
These keywords are often single words that describe the context of the site from which the word cloud is created.
A Word Cloud is formed by grouping these words together. The font size and color tone of each word in this cloud vary.
As a result, this depiction aids in determining terms of significance.
A larger font size for a word emphasizes its significance in comparison to other words in the cluster.
Word Clouds may be made in a variety of forms and sizes, depending on the creator's concept.
When building a Word Cloud, the amount of words is quite crucial.
More words do not necessarily equal a better Word Cloud since it gets cluttered and difficult to read.
A Word Cloud must always be semantically meaningful and reflect what it is intended to convey.

 ![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Capture2.PNG)

   **Training the ‘Skip-gram’ model**
   
   
Word embedding is a term used in natural language processing to represent words for text analysis, often in real-valued vectors that store the meaning of the word. Closed words in vector space are supposed to have comparable meanings.
Word embedding employs language modeling and feature learning methods to map vocabulary words to vectors of real numbers. 
Word2vec is a two-layer neural network that "vectorizes" words to analyze text.
It takes a text corpus as input and returns a collection of vectors as output.
Feature vectors corresponding to words in that corpus.

Once trained, a model of this kind may find synonymous terms or propose extra words for an incomplete text.
Word2vec, as the name indicates, represents each different word with a specific collection of integers known as a vector.
The vectors are carefully designed in such a way that a simple mathematical function (the cosine similarity between the vectors) reflects the amount of semantic similarity between the words represented by those vectors. Word2vec is a combination of CBOW(Continues Bag of word) and Skip-Gram. The continuous skip-gram model learns by predicting the words around the current word.
In other words, the Continuous Skip-Gram Model predicts terms before and after the current word in the same sentence. We train the genism word2vec model with our own custom corpus as following:

  ```
# training the model
skipgram = Word2Vec(x, vector_size =100, window = 1, min_count=2,sg = 1)
print(skipgram)
skipgram.save('skipgramx11.bin')
 ```
- x: The list of split sentences. 
- vector_size: The number of dimensions of the embeddings.
- window: The greatest distance between a target word and the words surrounding it. 
- min_count: The minimum number of words to consider while training the model; words with less than this number of occurrences will be disregarded..
- sg: CBOW(0) or skip gram(1) as the training algorithm. CBOW is the default training algorithm. 



    **Training the ‘FastText’ model**
    
 FastText is a Word2Vec enhancement suggested by Facebook in 2016.
FastText divides words into many n-grams rather than putting individual words into the Neural Network (sub-words).
We will get word embeddings for all n-grams given the training dataset after training the Neural Network.
Rare words can now be adequately represented since some of their n-grams are likely to exist in other words. 

```
FastText=FastText(x,vector_size=100, window=2, min_count=2, workers=5, min_n=1, max_n=2,sg=1)
FastText.save('FastText.bin')    #Saving our model
FastText = Word2Vec.load('FastText.bin')  #Loading our pretrained model
```
    
   **Model embeddings – Similarity**
   
   let’s try which words are most similar to the word “corona” and “patient” with Skig-gram .
   
   ![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Captureskip.PNG)
   
   let’s try which words are most similar to the word “lung” and “breathing” with FastText.
    
   ![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Capturefast.PNG)
   
   
   **PCA plots for Skip-gram and FastText models**

![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Capturepca.PNG)
![source](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Capturefastest.PNG)
    
   
  **Run the Streamlit Application**


 You can follow ![here](https://github.com/adrienpayong/Skip-Gram-Model-to-Create-Word-Embeddings/blob/main/Medical%20Embeddings_Final.ipynb)


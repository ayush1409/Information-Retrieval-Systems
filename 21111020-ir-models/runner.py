import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

import re
import os
import numpy as np
import sys
import string
import pickle
from collections import defaultdict
from time import perf_counter
import math
import csv
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

file = "posting.pkl"
file_obj_new = open(file, 'rb')
posting = pickle.load(file_obj_new)
#print(len(posting))
#print(type(posting))

# retrieve the total_doc_count dictionary
file = "token_doc_count.pkl"
file_obj = open(file, 'rb')
token_doc_count = pickle.load(file_obj)
#print(len(token_doc_count))
#print(type(token_doc_count))

# retrieve the doc_word_count
file = "doc_word_count.pkl"
file_obj = open(file, 'rb')
doc_word_count = pickle.load(file_obj)
#print(len(doc_word_count))
#print(type(doc_word_count))

print('\nPosting lists created using the corpus......')

# idf dict: key(token) -> value(total docs / number of docs containing that token)
idf = defaultdict(int)
total_docs = len(posting.keys())

for word, doc_cnt in token_doc_count.items():
    if doc_cnt != 0:
        idf[word] = math.log10(total_docs/doc_cnt)

doc_list = list(doc_word_count.keys())

avg_doc_size = sum(list(doc_word_count.values())) / len(doc_word_count)

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

def BooleanRetrieval(query, n=None):
    # preprocess the query
    query = query.lower()
    query_tokens = word_tokenize(query)

    query_tokens = [porter.stem(word) for word in query_tokens if word != 'and']

    # get unique elements in the list
    query_tokens = list(set(query_tokens))
    #print(query_tokens)

    # retrieve the model
    rel_docs = set()
    for word in query_tokens:
        if len(posting[word]) == 0:
            return list()
        p = set([w[0] for w in posting[word]])
        if len(rel_docs) == 0:
            rel_docs = p
        else:
            rel_docs = rel_docs & p
    
    if n == None or n >= len(posting):     
        return list(rel_docs)
    return list(rel_docs)[:n]


def compute_tf_idf(query, n = None):
    # preprocess the query
    query = query.lower()
    
    query = re.sub(r'[^\w\s]', '', query)
    query = re.sub(r'[^\x00-\x7f]', "", query)
    query_tokens = word_tokenize(query)
    
    query_tokens = [word for word in query_tokens if not word in stop_words]
    
    query_tokens = [porter.stem(word) for word in query_tokens if word != 'and']
    

    # get unique elements in the list
    query_tokens = list(set(query_tokens))
    #print(query_tokens)
    
    # compute tf-idf score and retreive the document using cosine similarity
    cosine_sim = defaultdict(float)
    for q_token in query_tokens:
        for doc, count in posting[q_token]:
            cosine_sim[doc] += count * idf[q_token]
    
    tf_idf_list = list(cosine_sim.items())
    tf_idf_list.sort(key = lambda x: x[1], reverse=True)
    
    rel_docs = [word[0] for word in tf_idf_list]
    
    if n is None or n >= len(tf_idf_list):
        return rel_docs
    return rel_docs[:n]


def bm25_scores(query, n=None):

    k = 2
    b = 0.75
    
    # preprocess the query
    query = query.lower()
    query_tokens = word_tokenize(query)

    query_tokens = [porter.stem(word) for word in query_tokens if word != 'and']

    # get unique elements in the list
    query_tokens = list(set(query_tokens))
    
    # calculate the bm25 scores for each of the document
    bm25 = defaultdict(float)
    for q in query_tokens:
        for doc in doc_list:
            tf = [element[1] for element in posting[q] if element[0] == doc]
            
            # if the current document doesn't contain the query term, don't need to rank it
            if len(tf) == 0:
                continue
            if bm25[doc] == float(0):
                bm25[doc] = idf[q] * ((tf[0] * (k+1))/(tf[0] + k*(1 - b + b*avg_doc_size)))
            else:
                bm25[doc] += idf[q] * ((tf[0] * (k+1))/(tf[0] + k*(1 - b + b*avg_doc_size)))
                
    bm25_list = list(bm25.items())
    bm25_list.sort(key = lambda x : x[1], reverse=True)
    
    rel_docs = [word[0] for word in bm25_list]
    
    if n is None or n >= len(bm25_list):
        return rel_docs
    
    return rel_docs[:n]

print('\nAll Information Retrival models are created....')

file_path = sys.argv[1]

queries = []
with open(file_path) as file:
    query_set_file = csv.reader(file, delimiter="\t")
    for row in query_set_file:
        queries.append(row)

print('\nRetriveing top 5 documents for each of the models and saving them in QRels format(in seperate files)...')

# 1. loop over all the queries
# 2. run query in all 3 models, each returning top 3 document
# 3. Write the output document list in Qrels format in seperate 3 files

qrels_bool, qrels_tfidf, qrels_bm25 = [], [], []

wordnet = WordNetLemmatizer()

for query in queries[1:]:
    qid = query[0]
    text = query[1]
    # print("qid : {}, text : {}", query[0], query[1])
    # print(query)
    rel_docs_bool = BooleanRetrieval(text, n=5)
    for doc in rel_docs_bool:
        qrels_bool.append([qid, '1', doc, '1'])
        
    rel_docs_tfidf = compute_tf_idf(text, n=5)
    for doc in rel_docs_tfidf:
        qrels_tfidf.append([qid, '1', doc, '1'])
        
    rel_docs_bm25 = bm25_scores(text, n=5)
    for doc in rel_docs_bm25:
        qrels_bm25.append([qid, '1', doc, '1'])
        
qrels_df_bool = pd.DataFrame(qrels_bool)
qrels_df_bool.to_csv('QRels_boolean.csv', index=False, header=False)

qrels_df_tfidf = pd.DataFrame(qrels_tfidf)
qrels_df_tfidf.to_csv('QRels_tfidf.csv', index=False, header=False)

qrels_df_bm25 = pd.DataFrame(qrels_bm25)
qrels_df_bm25.to_csv('QRels_bm25.csv', index=False, header=False)

print('\nQRels_boolean.csv, QRels_tfidf.csv, QRels_bm25.csv files are generated....')


# - Vishal Mittal 2017A7PS0080P
# - Laksh Singla 2017A7PS0082P
# - Yash Vijay 2017A7PS0072P
#   Wikipedia files used: AB - wiki_00 to wiki_04 (5 files)

import numpy as np
import pandas as pd
import re
import string
import nltk
import os
import math
import pickle
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from collections import Counter
from spellchecker import SpellChecker

wordnet_lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

INDEX_FILE = 'index.pickle'
MODIFIED_INDEX_FILE = 'modified_index.pickle'

_idf = None
_term_doc_weights = None
_champion_list = None
_doc_id_title_mapping = None

_modified_idf = None
_modified_term_doc_weights = None
_modified_champion_list = None
_modified_doc_id_title_mapping = None

query = None

def load_index():
    global _idf
    global _term_doc_weights
    global _champion_list
    global _doc_id_title_mapping

    global _modified_idf
    global _modified_term_doc_weights
    global _modified_champion_list
    global _modified_doc_id_title_mapping

    with open(INDEX_FILE, 'rb') as f:
        # print('her')
        loaded_obj = pickle.load(f)
        _idf = loaded_obj['idf']
        _term_doc_weights = loaded_obj['term_doc_weights']
        _champion_list = loaded_obj['champion_list']
        _doc_id_title_mapping = loaded_obj['doc_id_title_mapping']

    with open(MODIFIED_INDEX_FILE, 'rb') as f:
        # print('her')
        loaded_obj = pickle.load(f)
        _modified_idf = loaded_obj['idf']
        _modified_term_doc_weights = loaded_obj['term_doc_weights']
        _modified_champion_list = loaded_obj['champion_list']
        _modified_doc_id_title_mapping = loaded_obj['doc_id_title_mapping']

def get_term_weights_for_query(query, idf, query_type = 0):

    raw_query_tokens = [token for token in word_tokenize(query) if token not in string.punctuation]
    query_terms = [token.lower() for token in raw_query_tokens]
    
    if query_type == 1:
      query_terms = [spell.correction(w) for w in query_terms]
      query_terms = [wordnet_lemmatizer.lemmatize(w) for w in query_terms]
      
    result = defaultdict(int)
    for term in query_terms:
        result[term] += 1

    for term in result:
        result[term] = (1 + math.log10(result[term])) * (idf.get(term, 0))

    vector_length = 0
    for weight in result.values():
        vector_length += weight ** 2
    vector_length = math.sqrt(vector_length)

    if vector_length == 0:
      vector_length = 1     # To avoid Divide by Zero errors

    # normalization
    for term in result:
        result[term] /= vector_length

    return query_terms, result

def search(query_type = 0):

    k = int(input("\nEnter No. of documents to retrieve: "))   


    if query_type == 0:
        idf = _idf
        term_doc_weights = _term_doc_weights
        doc_id_title_mapping = _doc_id_title_mapping
        champion_list = _champion_list

    elif query_type == 1:
        idf = _modified_idf
        term_doc_weights = _modified_term_doc_weights
        doc_id_title_mapping = _modified_doc_id_title_mapping
        champion_list = _modified_champion_list

    query_terms, term_query_weights = get_term_weights_for_query(query, idf, query_type)
    doc_id_score = defaultdict(int)

    if query_type == 1:
        champion_docs = set()
        for word in query_terms:
            champion_docs = champion_docs.union(set(champion_list[word]))

    for term, query_weight in term_query_weights.items():
        for doc_id, doc_weight in term_doc_weights[term].items():
            if query_type == 1 and doc_id not in champion_docs: continue
            doc_id_score[doc_id] += query_weight * doc_weight

    doc_score_pairs = [(doc_id, score) for doc_id, score in doc_id_score.items()]

    doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)

    k = min(k, len(doc_score_pairs))

    for pair in doc_score_pairs[:k]:
        doc_id = int(pair[0])
        title = doc_id_title_mapping[str(doc_id)]
        score = pair[1]
        print("score = {0:.4f}, document_id = {1}, title = {2}".format(score, doc_id, title))


def run_repl():
    global query
    query = input('Enter the query: ')

    print('\nOptions:')
    print('1. Normal search based on lnc.ltc scoring scheme')
    print('2. Improvement using spelling correction + lemmatization + champion_list')
    query_type = int(input('\nEnter your choice (either 1 or 2): '))
    query_type = query_type - 1
    search(query_type)

def main():
    load_index()
    run_repl()

if __name__ == '__main__':
    main()
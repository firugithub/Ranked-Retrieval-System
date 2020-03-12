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

WIKIS_FOLDER = './wikis'
CHAMPION_LIST_COUNT = 100
PICKLE_FILE = 'index.pickle'
MODIFIED_PICKLE_FILE = 'modified_index.pickle'

wordnet_lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# One time download
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Instead of only one wiki, added 5 wikis for better results
response = ""

entries = os.listdir(WIKIS_FOLDER)
for entry in entries:
  f = open(os.path.join(WIKIS_FOLDER, entry), "r", encoding='utf8')
  response = response + f.read().lower()

doc_id_title_mapping = {}
doc_dicts = []

term_frequency = {}     # Creating dictionary of term frequencies
inverted_index = {}     # Inverted index containing document-wise frequency
idf = dict()

champion_list = {}
champion_docs = set()

# query_type = 1      # query_type = 0 => No improvement, part-1
                    # query_type = 1 => Improvement spellchecker + query_lemmatize + champion_list

"""## Parsing the text"""

def preprocess(query_type = 0):
  soup = BeautifulSoup(response, 'html.parser')
  TAG_RE = re.compile(r'<[^>]+>')

  all_docs = soup.find_all('doc')

  for doc in all_docs:
    doc_contents = TAG_RE.sub('', ''.join(map(lambda x: str(x), doc.contents)))

    doc_contents = doc_contents.translate(str.maketrans('', '', string.punctuation))  # Remove punctuations from the doc_contents
    doc_contents = doc_contents.replace("\n", " ")                                    # Remove unnecessary newlines
    doc_contents = ''.join(i for i in doc_contents if ord(i)<128)
    doc_contents = " ".join(doc_contents.split())
    
    if query_type == 1:
      doc_contents = word_tokenize(doc_contents)
      doc_contents = [wordnet_lemmatizer.lemmatize(w) for w in doc_contents]
      doc_contents = ' '.join(doc_contents)

    doc_dict = {
      'id': doc['id'],
      'url': doc['url'],
      'title': doc['title'],
      'content': doc_contents
    }

    doc_id_title_mapping[doc['id']] = doc['title']
    doc_dicts.append(doc_dict)

"""## Build the Index - Inverted Index Construction"""

def build_index(query_type = 0):
  for doc_dict in doc_dicts:
    print("Building_index for doc_id: {0}".format(doc_dict['id']))
    for word in word_tokenize(doc_dict['content']):    
      if word in term_frequency:
        term_frequency[word] = term_frequency[word] + 1
      else:
        term_frequency[word] = 1

      if word in inverted_index:
        posting_list = inverted_index[word]
        if doc_dict['id'] in posting_list:
          posting_list[doc_dict['id']] = posting_list[doc_dict['id']] + 1
        else:
          posting_list[doc_dict['id']] = 1
      else:
        inverted_index[word] = {doc_dict['id']:1}


def get_term_document_weights(inverted_index):
  documents_count = len(doc_dicts)
  document_length = defaultdict(int)  # Used as normalization factor (Cosine Similarity)

  term_doc_weights = defaultdict(dict)

  for term, posting_list in inverted_index.items():
      idf[term] = math.log10(documents_count / len(posting_list))
      for doc_id, tf in posting_list.items():
          weight = 1 + math.log10(tf)
          term_doc_weights[term][doc_id] = weight
          document_length[doc_id] += weight ** 2

  # Use sqrt of weighted square distance for cosine normalization
  for doc_id in document_length:
      document_length[doc_id] = math.sqrt(document_length[doc_id])

  # normalization
  for term in term_doc_weights:
      for doc_id in term_doc_weights[term]:
          term_doc_weights[term][doc_id] /= document_length[doc_id]

  return term_doc_weights

def create_champion_list():
    for word in term_frequency:
        posting_list = inverted_index[word]
        c = Counter(posting_list)
        mc = c.most_common(min(CHAMPION_LIST_COUNT, len(posting_list)))
        most_common_docs = [i[0] for i in mc]
        champion_list[word] = most_common_docs


def main():
    global inverted_index
    global term_frequency
    global idf
    global doc_id_title_mapping
    global champion_list

    inverted_index = {}
    term_frequency = {}     
    idf = dict()
    doc_id_title_mapping = {}
    champion_list = {}

    query_type = 0
    print("\nBuilding_index for normal search")

    preprocess(query_type)
    build_index(query_type)
    term_doc_weights = get_term_document_weights(inverted_index)
    create_champion_list()
    pickeled_obj = {}
    pickeled_obj['inverted_index'] = inverted_index
    pickeled_obj['term_frequency'] = term_frequency
    pickeled_obj['idf'] = idf
    pickeled_obj['term_doc_weights'] = term_doc_weights
    pickeled_obj['doc_id_title_mapping'] = doc_id_title_mapping
    pickeled_obj['champion_list'] = champion_list
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(pickeled_obj, f)


    inverted_index = {}
    term_frequency = {}     
    idf = dict()
    doc_id_title_mapping = {}
    champion_list = {}



    query_type = 1
    print("\nBuilding_index for improved search (after lemmatization)")

    preprocess(query_type)
    build_index(query_type)
    term_doc_weights = get_term_document_weights(inverted_index)
    create_champion_list()
    pickeled_obj_modified = {}
    # pickeled_obj_modified['inverted_index'] = inverted_index
    # pickeled_obj_modified['term_frequency'] = term_frequency
    pickeled_obj_modified['idf'] = idf
    pickeled_obj_modified['term_doc_weights'] = term_doc_weights
    pickeled_obj_modified['doc_id_title_mapping'] = doc_id_title_mapping
    pickeled_obj_modified['champion_list'] = champion_list
    with open(MODIFIED_PICKLE_FILE, 'wb') as f:
        pickle.dump(pickeled_obj_modified, f)

    print('Index created')

if __name__ == '__main__':
    main()
# IR Assignment-2: Vector Space Based Ranked Retrieval System

An IR system is built with the following characteristics:
1. The vector space model is used for computing the score between document and query.
2. `lnc.ltc` scoring scheme (based on SMART notation) is implemented for scoring.
3. The system supports free-text queries.

## The following packages need to be installed

- nltk
- bs4
- collections
- pyspellchecker
- pickle
- numpy
- requests

## Structure

There are 2 functionalities:

1. index_builder
2. search

### index_builder

This file creates an inverted index for documents in the folder `wikis` and stores it in a pickle file for further processing.

For lemmatization, WordNetLemmatizer from nltk is used.

### search

It takes as an input a query and gives as output the top K documents. This file never reads the text corpus.

## User Interface

Enter the query: number theory

Options:
1. Normal search based on lnc.ltc scoring scheme
2. Improvement using spelling correction + lemmatization + champion_list

Enter your choice (either 1 or 2): 1

Enter No. of documents to retrieve: 10
score = 0.1452, document_id = 174754, title = teiji takagi
score = 0.1227, document_id = 172199, title = faltings's theorem
score = 0.1215, document_id = 180839, title = victor vroom
score = 0.1153, document_id = 174724, title = goro shimura
score = 0.1102, document_id = 174448, title = abelian extension
score = 0.1084, document_id = 174108, title = abc conjecture
score = 0.1081, document_id = 178236, title = surautomatism
score = 0.1069, document_id = 181441, title = canterbury college
score = 0.1067, document_id = 177770, title = human development theory
score = 0.1058, document_id = 171537, title = planck temperature

## Implementation Details

For lemmatization, WordNetLemmatizer from nltk is used. Weighting scheme for
ranked retrieval is lnc.ltc:

1. Before asking any queries the system pre-calculates term-document weights, using the formula 1 + log10(term_frequency) and normalizes it by document vector's length (for cosine similarity). Results are stored in a Dictionary for fast future accesses. Also, inverse document frequency (idf) is computed for all terms.

2. When free-text query is typed, the system computes term-query weights using formula (1 + log10(term_frequency_in_query)) * idf(term) and normalizes them. It requires linear time depending on the query length.

3. To efficiently calculate document scores, term-at-a-time approach (bag of words) is used for query terms:
```
	for term, query_weight in term_query_weights.items():
		for doc_id, doc_weight in term_doc_weights[term].items():
			doc_id_score[doc_id] += query_weight * doc_weight
```
Time complexity will linearly depend on the number of term-document pairs for query terms.

4. Documents are sorted by their scores in (O(N log N), where N is the number of documents, containing query terms) to show top relevant.

## Guidelines to run the assignments

1. Build the index (Needs to be done only once and it will create `index.pickle` and `modified_index.pickle` files which would store inverted_index and other relevant information)

```
python3 build_index.py
```

2. To search any query:

```
python3 test_queries.py
```


### Group Members

- Vishal Mittal [Github](https://github.com/vismit2000)
- Laksh Singla [Github](https://github.com/LakshSingla)
- Yash Vijay


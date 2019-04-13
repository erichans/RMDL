"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: kk7nc@virginia.edu

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import spacy
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger('text_feature_extraction')

stopwords = set(stopwords.words('portuguese'))
nlp = spacy.load('pt_core_news_sm')
stopwords |= nlp.Defaults.stop_words

def transliterate(line):
    cedilla2latin = [[u'Á', u'A'], [u'á', u'a'], [u'Č', u'C'], [u'č', u'c'], [u'Š', u'S'], [u'š', u's']]
    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line








def text_cleaner(text,
                 deep_clean=False,
                 stem= True,
                 stop_words=True,
                 translite_rate=True):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning

    ]

    if deep_clean:
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", " ")
        text = text.replace("=", " ")
        text = text.replace("?", " ")
        text = text.replace("!", " ")

        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        text = text.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        if translite_rate:
            text = transliterate(text)
        if stem:
            text = PorterStemmer().stem(text)
        text = WordNetLemmatizer().lemmatize(text)
        if stop_words:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(str(e) for e in text)
    else:
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
    return text.lower()

def loadData_Tokenizer(X_train, X_test,GloVe_DIR,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)
    logger.info(f'Tokenizing {MAX_NB_WORDS} tokens')
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    logger.info(f'fit_on_texts...')
    tokenizer.fit_on_texts(text)
    logger.info(f'texts_to_sequences...')
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    logger.info(f'pad_sequences...')
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    logger.info('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    logger.info(text.shape)
    X_train = text[0:len(X_train), ]
    X_test = text[len(X_train):, ]
    
    word2vec_model = KeyedVectors.load_word2vec_format(GloVe_DIR, binary=True)
    embeddings_index = word2vec_model.wv
    del word2vec_model

    logger.info('Total %s word vectors.' % len(embeddings_index.vocab))
    return (X_train, X_test, word_index,embeddings_index)

def loadData(X_train, X_test,MAX_NB_WORDS=75000):
    logger.info(f'TfidfVectorizer {MAX_NB_WORDS} tokens')
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    logger.info(f'fit_transform x_train...')
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    logger.info(f'transform x_test...')
    X_test = vectorizer_x.transform(X_test).toarray()
    logger.info(f"tf-idf with {str(np.array(X_train).shape[1])} features")
    return (X_train,X_test)

#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import re
import itertools
import pickle
import jieba
import codecs
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def save_vocabulary(vocabularyfile, vocabulary, vocabulary_inv):
    v_file = file(vocabularyfile,"wb")
    pickle.dump(vocabulary, v_file)
    pickle.dump(vocabulary_inv, v_file)
    v_file.close()

def restore_vocabulary(vocabularyfile):
    v_file = file(vocabularyfile, "rb")
    vocabulary = pickle.load(v_file)
    vocabulary_inv = pickle.load(v_file)
    v_file.close()
    return vocabulary,vocabulary_inv

def load_test_data(test_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    test_examples = list(codecs.open(test_data_file, "r", 'utf-8').readlines())
    test_examples = [s.strip() for s in test_examples]
    return [list(jieba.cut(sent)) for sent in test_examples]

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open(positive_data_file, "r", 'utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(codecs.open(negative_data_file, "r", 'utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [list(jieba.cut(sent)) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
  """
  Pads all sentences to the same length. The length is defined by the longest sentence.
  Returns padded sentences.
  """
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences

def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = np.array(labels)
  return [x, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

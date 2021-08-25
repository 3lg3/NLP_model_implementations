#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow
import string
import gensim
import transformers

from typing import List



def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()


def get_candidates(lemma, pos):
    # Part 1
    possible_subs = set()
    for lexeme in wn.lemmas(lemma, pos):
        for l in lexeme.synset().lemmas():
            sub = l.name().replace("_", " ")
            if sub != lemma:
                possible_subs.add(sub)

    return possible_subs


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'


def wn_frequency_predictor(context: Context) -> str:
    candidates = get_candidates(context.lemma, context.pos)
    word_freq = {}  # key: word , value: count
    # initialize the dict
    for candidate in candidates:
        word_freq[candidate] = 0
    #
    for lexeme in wn.lemmas(context.lemma, context.pos):
        for l in lexeme.synset().lemmas():
            sub = l.name().replace("_", " ")
            if sub in word_freq.keys():
                word_freq[sub] += l.count()

    return max(word_freq, key=word_freq.get)


def wn_simple_lesk_predictor(context: Context) -> str:
    context_toks = set()
    context_toks.union(context.left_context, context.right_context)
    context_toks -= set(stopwords.words('english'))
    overlap = {}
    for lexeme in wn.lemmas(context.lemma, context.pos):
        synset = lexeme.synset()
        overlap[synset] = 0
        # definition of synset
        for w1 in tokenize(synset.definition()):
            if w1 in context_toks:
                overlap[synset] += 1
        # examples of synset
        for example in synset.examples():
            for w2 in tokenize(example):
                if w2 in context_toks:
                    overlap[synset] += 1
        # hypernyms of synset
        for hyper in synset.hypernyms():
            for w3 in tokenize(hyper.definition()):
                if w3 in context_toks:
                    overlap[synset] += 1
            for example in hyper.examples():
                for w4 in tokenize(example):
                    if w4 in context_toks:
                        overlap[synset] += 1

    # select the synset(s) with max overlap
    target_freq = {}
    for synset, count in overlap.items():
        if count == max(overlap.values()):
            target_freq[synset] = 0
            for l in synset.lemmas():
                sub = l.name().replace("_", " ")
                if sub == context.lemma:
                    target_freq[synset] += l.count()

    # Then select with the highest target lexemes frequency if there is a tie or no overlap
    chosen = max(target_freq, key=target_freq.get)

    # select the candidate lexeme (other than target word) with highest frequency
    candidate_freq = {}
    for l in chosen.lemmas():
        sub = l.name().replace("_", " ")
        if sub != context.lemma:
            candidate_freq[sub] = l.count()

    if len(candidate_freq) == 0:
        return context.lemma

    return max(candidate_freq, key=candidate_freq.get)


class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        max_sim = -float("inf")
        nearest = None
        for syn in candidates:
            if syn in self.model.vocab:
                sim = self.model.similarity(syn, context.lemma)
                if sim > max_sim:
                    max_sim = sim
                    nearest = syn

        return nearest  # replace for part 4


class BertPredictor(object):

    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        mask_index = len(context.left_context) + 1
        concat_tokens = ["[MASK]"]
        concat_tokens = context.left_context + concat_tokens + context.right_context
        # sentence = ' '.join(concat_tokens)
        input_toks = self.tokenizer.encode(concat_tokens)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_index])[::-1]
        pred_list = self.tokenizer.convert_ids_to_tokens(best_words)
        best_syn = None
        min_index = float("inf")
        for syn in candidates:
            if syn in pred_list:
                index = pred_list.index(syn)
                if index < min_index:
                    min_index = index
                    best_syn = syn

        # if best_syn is None:
        #     return context.lemma

        return best_syn


# part 6: a simplified version of Continuous bag-of-words (CBOW)
# Input: context, window_size (the sie of context window that we want represent)
# Implementation Explanation: In part 4, we only represent the context according to the target word.
# However, in part 5, BERT (with the mask) only take the left context and right context and replace the target word
# with '[MASK]'
# Therefore, for part 6 here, I am going to improve the Word2Vec by represent the context based on both the word
# and its context (right and left) with the input window_size, which is similar to CBOW.
# Here, since it is not easy to obtain a weight matrix for context words, I decide to create my own weight matrix:
# the target word has the weight of 1 ;
# the closest words(left/right neighbor) to the target word has the highest weight (= 0.8)
# the weight decreases evenly based on the distance from the word to target word.
class Word2VecCBOW(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def cbow_predict(self, context: Context, window_size) -> str:
        # get the context of right window and right window with window_size
        left = []
        left += context.left_context[-window_size:]
        right = []
        right += context.right_context[:window_size]
        # remove stopwords and replace with mark '[STOP]'
        for word in left:
            if word in stopwords.words('english'):
                left[left.index(word)] = '[STOP]'

        for word in right:
            if word in stopwords.words('english'):
                right[right.index(word)] = '[STOP]'

        # create a weight matrix for vector representation of words.
        # Here the closest words to target have the highest weight of 1,
        # and decreasing by a gap (calculated by window_size) corresponding to distance from target
        gap = 0.8 / window_size
        weight = []
        curr_weight = 0
        for i in range(window_size):
            curr_weight += gap
            weight.append(curr_weight)

        for i in range(window_size):
            weight.append(curr_weight)
            curr_weight -= gap

        # create a new vector representation for the target based on its context window
        window = left + right
        window_vector = np.copy(self.model.wv[context.lemma])
        index = 0
        for word in window:
            if word in self.model.vocab and word != '[STOP]':
                window_vector += weight[index] * self.model.wv[word]
            index += 1

        candidates = get_candidates(context.lemma, context.pos)
        max_sim = -float("inf")
        best = None
        for syn in candidates:
            if syn in self.model.vocab:
                sim = self.cos(self.model.wv[syn], window_vector)
                if sim > max_sim:
                    max_sim = sim
                    best = syn

        return best

    def cos(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    # print("main")
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)
    # predictor = BertPredictor()
    predictor = Word2VecCBOW(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        # prediction = predictor.predict(context)
        prediction = predictor.cbow_predict(context, 5)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

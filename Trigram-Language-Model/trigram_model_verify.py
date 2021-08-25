import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np



def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    # unigrams does not need the 'START' token
    if n >= 2:
        sequence = ['START'] * (n - 1) + sequence
    sequence += ['STOP']
    # print(sequence)
    ngrams = []
    ngrams_count = len(sequence) - n + 1
    for i in range(ngrams_count):
        t = tuple(sequence[i:i + n])
        ngrams.append(t)
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        self.sentencecount = 0
        for sentence in corpus:
            self.sentencecount += 1
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1
            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1
            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # Check the length of trigram
        assert len(trigram) == 3, "Invalid trigram: trigram should be tuple with size 3."

        # trigram (a, b, c)

        # if trigram is ('START', 'START', c), it's edge case because there is no ['START','START'] in bigrams.
        if trigram[:2] == ('START', 'START'):
            return self.trigramcounts[trigram] / self.sentencecount

        # If there is no bigram (a,b), return unigram prob for c
        if self.bigramcounts[trigram[:2]] == 0:
            return self.raw_unigram_probability(trigram[2:])

        # Normal cases with denominator NOT equal to 0 and trigram NOT starting with two 'START'
        return self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # Check the length of bigram
        assert len(bigram) == 2, "Invalid bigram: bigram should be tuple with size 2."
        # bigram (a, b)

        # if bigram is ('START', b). (There is no ['START',] in unigrams.)
        if bigram[:1] == ('START',):
            return self.bigramcounts[bigram] / self.sentencecount

        # Normal cases with bigram NOT starting with 'START'
        # Denominator will NOT be zero here because there are no unseen words (as corpus reader replace words that are
        # not in the lexicon with UNK.
        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        # Check the length of unigram
        assert len(unigram) == 1, "Invalid unigram: bigram should be tuple with size 1."
        # unigram (a,)

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        if not hasattr(self, 'wordcount'):
            self.wordcount = sum(self.unigramcounts.values())

        return self.unigramcounts[unigram] / self.wordcount

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = list()
        bigram = ('START', 'START')
        for i in range(t):
            trigrams = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == bigram]
            probs = [self.raw_trigram_probability(trigram) for trigram in trigrams]
            pred = np.random.choice([trigram[2] for trigram in trigrams], 1, p=probs)[0]
            bigram = (bigram[1], pred)
            result.append(pred)
            if pred == 'STOP':
                return result

        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        return (lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(trigram[1:])
                + lambda3 * self.raw_unigram_probability(trigram[2:]))

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        logprob = 0
        for trigram in trigrams:
            logprob += math.log2(self.smoothed_trigram_probability(trigram))

        return logprob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        logprob_sum = 0
        nwords = 0
        for sentence in corpus:
            logprob_sum += self.sentence_logprob(sentence)
            nwords += len(sentence)

        return 2 ** (-logprob_sum / nwords)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        correct += (pp1 < pp2)
        total += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        # ..
        correct += (pp2 < pp1)
        total += 1

    return correct / total


if __name__ == "__main__":
    print("Running the main function...\n")
    model = TrigramModel("hw1_data/brown_train.txt")
    dev_corpus = corpus_reader("hw1_data/brown_test.txt", model.lexicon)
    train_corpus = corpus_reader("hw1_data/brown_train.txt", model.lexicon)
    pp_test = model.perplexity(dev_corpus)
    pp_train = model.perplexity(train_corpus)
    acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt",
                                   "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print("---------------  My Output: --------------------")
    print("Perplexity of brown_train: ", pp_train)
    print("Perplexity of brown_test: ", pp_test)
    print("Accuracy of the prediction: ", acc)
    print("------------------------------------------------")

    # ---------------  My Output: --------------------
    # Perplexity of brown_train:  18.061550189741123
    # Perplexity of brown_test:  281.3869286678916
    # Accuracy of the prediction:  0.8426294820717132
    # ------------------------------------------------

    # --- original main function --- #
    # model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

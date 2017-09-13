import math
import nltk
import random
from scipy.optimize import minimize_scalar
from collections import Counter

class NGramStorage:
    """Storage for ngrams' frequencies.
    
    Args:
        sents (list[list[str]]): List of sentences from which ngram
            frequencies are extracted.
        max_n (int): Upper bound of the length of ngrams.
            For instance if max_n = 2, then storage will store
            0, 1, 2-grams.
            
    Attributes:
        max_n (Readonly(int)): Upper bound of the length of ngrams.
    """
        
    def __init__(self, sents=[], max_n=0):
        self.__max_n = max_n
        self.__ngrams = {i: Counter() for i in range(self.__max_n + 1)}
        # self._ngrams[K] should have the following interface:
        # self._ngrams[K][(w_1, ..., w_K)] = number of times w_1, ..., w_K occured in words
        # self._ngrams[0][()] = number of all words
        
        ### YOUR CODE HERE
        self.__ngrams[0][()] = sum(len(sents[i]) for i in range(len(sents)))
    
        for n in range(1, self.__max_n + 1):
            for sentence in sents:
                for ngram in nltk.ngrams(sentence, n):
                    self.__ngrams[n][ngram] += 1

        ### END YOUR CODE
        
    def add_unk_token(self):
        """Add UNK token to 1-grams."""
        # In order to avoid zero probabilites 
        if self.__max_n == 0 or u'UNK' in self.__ngrams[1]:
            return
        self.__ngrams[0][()] += 1
        self.__ngrams[1][(u'UNK',)] = 1
        
    @property
    def max_n(self):
        """Get max_n"""
        return self.__max_n
        
    def __getitem__(self, k):
        """Get dictionary of k-gram frequencies.
        
        Args:
            k (int): length of returning ngrams' frequencies.
            
        Returns:
            Dictionary (in fact Counter) of k-gram frequencies.
        """
        # Cheking the input
        if not isinstance(k, int):
            raise TypeError('k (length of ngrams) must be an integer!')
        if k > self.__max_n:
            raise ValueError('k (length of ngrams) must be less or equal to the maximal length!')
        return self.__ngrams[k]
    
    def __call__(self, ngram):
        """Return frequency of a given ngram.
        
        Args:
            ngram (tuple): ngram for which frequency should be computed.
            
        Returns:
            Frequency (int) of a given ngram.
        """
        # Cheking the input
        if not isinstance(ngram, tuple):
            raise TypeError('ngram must be a tuple!')
        if len(ngram) > self.__max_n:
            raise ValueError('length of ngram must be less or equal to the maximal length!')
        if len(ngram) == 1 and ngram not in self.__ngrams[1]:
            return self.__ngrams[1][(u'UNK', )]
        return self.__ngrams[len(ngram)][ngram]

def perplexity(estimator, sents):
    '''Estimate perplexity of the sequence of words using prob_estimator.'''
    ### YOUR CODE HERE
    # Avoid log(0) by replacing zero by 10 ** (-50).
    sum_log_pr = 0
    sum_len = sum(len(sents[i]) for i in range(len(sents)))
    ZERO_BORDER = 10 ** (-50)
    for sent in sents:
        pr = estimator.prob(sent)
        if pr < ZERO_BORDER:
            pr = ZERO_BORDER
        sum_log_pr += math.log(pr)
    sum_log_pr *= (-1. / sum_len)
    perp = math.exp(sum_log_pr)
    ### END YOUR CODE
    
    return perp

class LaplaceProbabilityEstimator:
    """Class for probability estimations of type P(word | context).
    
    P(word | context) = (c(context + word) + delta) / (c(context) + delta * V), where
    c(sequence) - number of occurances of the sequence in the corpus,
    delta - some constant,
    V - number of different words in corpus.
    
    Args:
        storage(NGramStorage): Object of NGramStorage class which will
            be used to extract frequencies of ngrams.
        delta(float): Smoothing parameter.
    """
    
    def __init__(self, storage, delta=1.):
        self.__storage = storage
        self.__delta = delta
        
    def cut_context(self, context):
        """Cut context if it is too large.
        
        Args:
            context (tuple[str]): Some sequence of words.
        
        Returns:
            Cutted context (tuple[str]) up to the length of max_n.
        """
        if len(context) + 1 > self.__storage.max_n:
            context = context[-self.__storage.max_n + 1:]
        return context
        
    def __call__(self, word, context):
        """Estimate conditional probability P(word | context).
        
        Args:
            word (str): Current word.
            context (tuple[str]): Context of a word.
            
        Returns:
            Conditional probability (float) P(word | context).
        """
        # Cheking the input
        if not isinstance(word, str):
            raise TypeError('word must be a string!')
        if not isinstance(context, tuple):
            raise TypeError('context must be a tuple!')
            
        ### YOUR CODE HERE
        # If context is too large, let's cut it.
        context = self.cut_context(context)
        phrase_counts = self.__storage(context + (word, ))
        context_counts = self.__storage(context)
        prob = (1. * phrase_counts + self.__delta) / (context_counts + self.__delta * len(self.__storage[1]))
        ### END YOUR CODE
        
        return prob
    
    def prob(self, sent):
        """Estimate probability of a sentence using Markov rule.
        
        Args:
            sentence (list[str]): Sentence for probability estimation.
            
        Returns:
            Probability (float) P(sentence).
        """
        prob = 1.
        for i in range(len(sent)):
            prob *= self(sent[i], tuple(sent[:i]))
        return prob
    
# Try to find out best delta parametr. We will not provide you any strater code.
### YOUR CODE HERE
import numpy as np


def _laplace_perplexity(storage, test_sents, delta):
    laplace_estimator = LaplaceProbabilityEstimator(storage, delta)
    perp = perplexity(laplace_estimator, test_sents)
    return perp


def get_best_laplace_estimator_delta(sentences, n_for_ngram_storage=3, min_delta=0., max_delta=1000., seed=42):
    random.seed(seed)
    random.shuffle(sentences)
    print('Number of all sentences = {}'.format(len(sentences)))
    train_sents = sentences[:int(0.8 * len(sentences))]
    test_sents = sentences[int(0.8 * len(sentences)):]
    print('Number of train sentences = {}'.format(len(train_sents)))
    print('Number of test sentences = {}'.format(len(test_sents)))
    storage = NGramStorage(train_sents, n_for_ngram_storage)
    bounds = (0., 1000.)
    best_delta = minimize_scalar(lambda x: _laplace_perplexity(storage, test_sents, x), method = 'bounded', bounds = bounds,
                                 options={'xatol': 1e-3, 'disp': True}).x
    
    return best_delta

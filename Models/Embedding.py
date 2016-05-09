from keras.models import *
from keras.layers import *
import numpy as np
from collections import Counter

#TODO continue testing
    #random init seems to work
    #TODO test situation where we begin with initialized values
class Embedding_keras:
    """
    builds the embedding layer to be used and updated during training
    :param load_w2v If `true`, initialize values with existing embeddings
    :param gensim_class The instance of `gensim` word2vec to intialize values
    :param w2v_dimension The size of the embeddings
    :param W_regularizer
    :param W_constraint
    :param activity_regularizer
    :param mask_zero
    :param dropout
    """

    #eRandom = Embedding_keras(load_w2v=False, voc_size=1000, w2v_dimension=200)
    #eLoaded = Embedding_keras(load_w2v=True, gensim_class = w2v, voc_size=len(w2v.index2word), w2v_dimension=len(w2v["the"]))

    def __init__(self, load_w2v=False, gensim_class=None, w2vDimension=200, W_regularizer=None, W_constraint=None, activity_regularizer=None, mask_zero=True, dropout=0):
        #hyper-parameters
        # if mask_zero:
        #     self.voc_size = voc_size + 1
        # else:
        self.vocSize = None
        self.w2vDimension = w2vDimension
        self.init = "uniform"
        self.W_regularizer = W_regularizer
        self.W_constraint = W_constraint
        self.activity_regularizer = activity_regularizer
        self.mask_zero = mask_zero
        self.dropout = dropout
        self.weights = None
        #randomly initialize word vectors?  Or initialize with pretrained?
        self.load_w2v = load_w2v
        self.w2v = gensim_class
        #variable reserved for embedding layer
        self.layer = None


    #if load_w2v == False this must be run BEFORE build()
    def getVocabSize(self, fPath, num_lines):
        #open file
        f = open(fPath, "rb")
        #make counter to estimate voc size
        vocCounter = Counter()
        #make counter to keep track of number of lines to process
        line_counter = 0
        #estimate vocabulary size
        if num_lines != 0:
            for line in f:
                line_counter += 1
                if line_counter <= num_lines:
                    clean = line.rstrip()
                    tokens = clean.split(" ")
                    for t in tokens:
                        vocCounter[t] += 1
        #if num_lines is set to 0, use entire file
        else:
            for line in f:
                line_counter += 1
                clean = line.rstrip()
                tokens = clean.split(" ")
                for t in tokens:
                    vocCounter[t] += 1
        f.close()
        #set vocabulary size
        self.vocSize = len(vocCounter.items())


    def build(self):
        #if randomly initializing word embedding layer
        if not self.load_w2v:
            self.layer = embeddings.Embedding(
                    input_dim=self.vocSize,
                    output_dim=self.w2vDimension,
                    init="uniform",
                    input_length=1,
                    batch_input_shape=(1,1),            #handles one one-hot vector at a time
                    W_regularizer=self.W_regularizer,
                    W_constraint=self.W_constraint,
                    activity_regularizer=self.activity_regularizer,
                    mask_zero=self.mask_zero,
                    dropout=self.dropout,
                    weights=None
            )
            #if initializing word embedding layer with pre-trained vectors
            #using own word vectors instead of training them in model: https://github.com/fchollet/keras/issues/853
            #requires vocabulary index and vectors
        else:
            #vocabulary in a list
            #indices must be from 1!
            vocab = self.w2v.index2word

            #build embedding weights matrix
            if self.mask_zero:
                self.weights = np.zeros((self.voc_size + 1, self.w2vDimension))        #TODO confirm this is correct for masking
                #populate weights
                #TODO confirm this is correct for masking
                for i in range(len(vocab)):
                    self.weights[i + 1,:] = self.w2v[vocab[i]]       #populate each row in weight matrix with the pretrained vector
            else:
                self.weights = np.zeros((self.voc_size, self.w2vDimension))
                #populate weights
                for i in range(len(vocab)):
                    self.weights[i,:] = self.w2v[vocab[i]]           #populate each row in weight matrix with the pretrained vector



            #build layer
            self.layer = embeddings.Embedding(
                    input_dim=self.voc_size,
                    output_dim=self.w2vDimension,
                    input_length=1,
                    batch_input_shape=(1,1),
                    W_regularizer=self.W_regularizer,
                    W_constraint=self.W_constraint,
                    activity_regularizer=self.activity_regularizer,
                    mask_zero=self.mask_zero,
                    dropout=self.dropout,
                    weights=[self.weights]
            )







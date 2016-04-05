from keras.models import *
from keras.layers import *
import numpy as np

class Embedding_keras:
    """
    builds the embedding layer to be used
    """

    def __init__(self, loadW2V=False, gensimW2Vclass=None, vocabSize=None, w2vDimension=300, W_regularizer=None, W_constraint=None, activity_regularizer=None, mask_zero=False, input_length=None, dropout=0):
        #hyper-parameters
        self.vocabSize = vocabSize
        self.dimension = w2vDimension
        self.init = "uniform"
        self.W_regularizer = W_regularizer
        self.W_constraint = W_constraint
        self.activity_regularizer = activity_regularizer
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.dropout = dropout
        self.weights = None
        #randomly initialize word vectors?  Or initialize with pretrained?
        self.loadW2V = loadW2V
        self.gensimW2Vclass = None
        #variable reserved for embedding layer
        self.layer = None


    def build(self):
        #if randomly initializing word embedding layer
        if self.loadW2V:
            self.layer = embeddings.Embedding(
                    input_dim=self.vocabSize,
                    output_dim=self.w2vDimension,
                    init="uniform",
                    input_length=self.input_length,
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
            vocab = self.gensimW2Vclass.index2word

            #build embedding weights matrix
            self.weights = np.zeros((self.vocabSize, self.dimension))       #TODO confirm this is correct for masking

            #populate weights
            #TODO confirm this is correct for masking
            for i in vocab:
                self.weights[i + 1,:] = self.gensimW2Vclass[vocab[i]]       #populate each row in weight matrix with the pretrained vector

            #build layer
            self.layer = embeddings.Embedding(
                    input_dim=self.vocabSize,
                    output_dim=self.dimension,
                    init="uniform",
                    input_length=self.input_length,
                    W_regularizer=self.W_regularizer,
                    W_constraint=self.W_constraint,
                    activity_regularizer=self.activity_regularizer,
                    mask_zero=self.mask_zero,
                    dropout=self.dropout,
                    weights=self.weights
            )




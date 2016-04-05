from keras.models import *
from keras.layers import *
import numpy as np

class LSTM_keras:
    """
    builds an LSTM using keras front end: http://keras.io/examples/#sequence-classification-with-lstm
    :param embeddingLayer: Embedding class (see above) if vectors are to be learned
    """

    def __init__(self, embeddingLayer=None, vocSize=None, w2vDimension=300, ):
        self.model = Sequential()

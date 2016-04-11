from keras.models import *
from keras.layers import *
import numpy as np

class LSTM_keras:
    """
    builds an LSTM using keras front end: http://keras.io/examples/#sequence-classification-with-lstm
    :param embeddingLayer: Embedding_keras class (see Embedding) if vectors are to be learned
    """

    # Input shape
    #3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    def __init__(self, embeddingLayer=None, vocSize=None, w2vDimension=None, cSize=300, max_seq_length=30, activation="tanh", inner_activation="hard_sigmoid", W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0, dropout_U=0):
        self.model = Sequential()
        if embeddingLayer:
            #get the dimensions from the embedding layer
            self.w2vDimension = embeddingLayer.w2vDimension
            #build the embedding layer
            self.model.add(embeddingLayer)
        else:
            #get the dimensions from the arguments
            #TODO what is this used for?
            self.w2vDimension = w2vDimension
        self.model.add(LSTM(
                            # input_shape=(None,w2vDimension),              #can this be None?
                            input_shape=(max_seq_length, w2vDimension),     #should allow for different batch sizes and required when the first layer of an architecture
                            output_dim=vocSize,
                            activation=activation,
                            inner_activation=inner_activation,
                            W_regularizer=W_regularizer,
                            U_regularizer=U_regularizer,
                            b_regularizer=b_regularizer,
                            dropout_W=dropout_W,
                            dropout_U=dropout_U,
                            return_sequences=False           #True when using more than one layer
                            ))
        # self.model.add(Dropout)
        self.model.add(Dense(vocSize))
        self.model.add(Activation("softmax"))

    # def train

"""
#only generates cost at end of sequence
#can only generate a prediction for word at N + 1 (where N is length of sentence)
>>> from keras.models import Sequential
Using TensorFlow backend.
>>> from keras.layers import LSTM, Dense
>>> import numpy as np
>>> w2v_dimension=8             #dimension of the input vector
>>> max_sentence_length=20      #length of each sequence -- HOW TO MAKE VARIABLE?
>>> voc_size=15                 #dimension of output of softmax layer
>>> c_size=5                    #dimension of the c state

>>> model = Sequential()
>>> model.add(LSTM(c_size, return_sequences=False, input_shape=(max_sentence_length, w2v_dimension)))
>>> model.add(Dense(voc_size, activation="softmax"))
>>> model.compile(loss="binary_crossentropy", optimizer="rmsprop")

#training data
#each training instance is a matrix where each row is a word in the sentence
#length_of_sentence x length_of_w2v
>>> x_train = np.random.random((1000, max_sentence_length, w2v_dimension))
>>> y_train=np.random.random((1000, voc_size))

#testing data
>>> x_val = np.random.random((100, max_sentence_length, w2v_dimension))
>>> y_val=np.random.random((100, voc_size))

>>> model.fit(x_train,y_train,batch_size=1,nb_epoch=1,show_accuracy=True,validation_data=(x_val,y_val))

#give it a sequence to predict next word
#can only predict word for N + 1 (where N is length of sentence)
>>> model.predict(x_val[0:1])   #has shape (batch_size, length_of_sentence, length_of_w2v)
#returns softmax across vocabulary for next word
array([[ 0.06752004,  0.06681924,  0.06783251,  0.06644987,  0.06461859,
         0.06652912,  0.06606538,  0.06768672,  0.06832125,  0.06933279,
         0.06447952,  0.06615783,  0.06422815,  0.06747453,  0.06648442]])

#stateful
#I think what I want, but not sure how to make it work
https://www.reddit.com/r/MachineLearning/comments/3dqdqr/keras_lstm_limitations/
>>> from keras.layers import LSTM, Dense
>>> import numpy as np
>>> w2v_dimension=8
>>> max_sentence_length=20
>>> voc_size=15
>>> c_size=5
>>> model=Sequential()
>>> model.add(LSTM(c_size, return_sequences=False, stateful=True,batch_input_shape=(1,max_sentence_length, w2v_dimension)))
I tensorflow/core/common_runtime/local_device.cc:40] Local device intra op parallelism threads: 8
I tensorflow/core/common_runtime/direct_session.cc:58] Direct session inter op parallelism threads: 8
>>> model.add(Dense(voc_size, activation="softmax"))
>>> model.compile(loss="binary_crossentropy", optimizer="rmsprop")

example: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

http://keras.io/faq/#how-can-i-use-stateful-rnns

X # this is our input data, of shape (32, 21, 16)   (batch size, length of sequence, vector size)
# we will feed it to our model in sequences of length 10
model = Sequential()
model.add(LSTM(32, batch_input_shape=(32, 10, 16), stateful=True))
model.add(Dense(16, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))
# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))
# let's reset the states of the LSTM layer:
model.reset_states()
# another way to do it in this case:
model.layers[0].reset_states()

"""
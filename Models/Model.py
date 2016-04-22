from keras.models import *
from keras.layers import *
import numpy as np
from random import shuffle
from collections import Counter
import Utils.PreProcessing as pre
import Utils.Evaluation as e
import Data as d

class LSTM_keras_LM:
    """
    builds an LSTM for language modeling using keras front end: http://keras.io/examples/#sequence-classification-with-lstm
    :param embeddingLayer Embedding_keras class (see Embedding) if vectors are to be learned
    :param embeddingClass `gensim` class of `word2vec`
    :param vocSize Size of vocabulary, will be dimensions of output
    :param w2vDimension Dimension of word embeddings
    :param cSize Size of cell state
    :param max_seq_length Largest sequence that will be handled; any sequences larger than this will be ignored
    :param W_regularizer ???
    :param U_regularizer ???
    :param b_regularizer ???
    :param dropout_W Percentage of nodes in W matrix to be dropped out
    :param dropout_U Percentage of nodes in U matrix (recurrent) to be dropped out
    :param loss_function Loss function to be used
    :param optimizer Optimizer to be used
    :param num_epochs Number of epochs
    """

    # Input shape
    #3D tensor with shape `(nb_samples, timesteps, input_dim)`.

    def __init__(self,
                 embeddingLayer=None,
                 embeddingClass=None,
                 vocSize=None,
                 w2vDimension=None,
                 cSize=300,
                 max_seq_length=30,
                 W_regularizer=None,
                 U_regularizer=None,
                 b_regularizer=None,
                 dropout_W=0,
                 dropout_U=0,
                 loss_function="categorical_crossentropy",
                 optimizer="rmsprop",
                 num_epochs=5,
                 training_vector=None,
                 testing_vector=None
                 ):
        self.embeddingLayer = embeddingLayer
        self.embeddingClass = embeddingClass
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.training_vectors = training_vector
        self.testing_vectors = testing_vector
        self.data = None
        self.vocSize = vocSize
        self.model = Sequential()
        if embeddingLayer:
            #get the dimensions from the embedding layer
            self.w2vDimension = embeddingLayer.w2vDimension
            #build the embedding layer
            self.model.add(embeddingLayer)
            #note: masking layer not needed here as it is handled in embedding layer
        else:
            #get the dimensions from the arguments
            #TODO what is this used for?
            self.w2vDimension = w2vDimension
            #manually add a masking layer (to handle variable length of sequences)
                #http://keras.io/layers/core/#masking
            self.model.add(Masking(mask_value=np.zeros(self.w2vDimension), batch_input_shape=(1,1,self.w2vDimension)))
        #add the LSTM layer
        self.model.add(LSTM(
                            output_dim=cSize,
                            activation="tanh",
                            inner_activation="hard_sigmoid",
                            W_regularizer=W_regularizer,
                            U_regularizer=U_regularizer,
                            b_regularizer=b_regularizer,
                            dropout_W=dropout_W,
                            dropout_U=dropout_U,
                            return_sequences=False,           #True when using more than one layer
                            stateful=True,
                            batch_input_shape=(1,1,self.w2vDimension)
                            ))
        # self.model.add(Dropout)
        self.model.add(Dense(
                                output_dim=vocSize,
                                activation="softmax"
                            ))
        self.model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
        self.model.summary()


    #file = file to use for training
    #num_lines = number of lines to use from training file
        #0 = all
    #training_vector = vector of training instances
    def train(self, fPath, num_lines):
        if not self.embeddingLayer:
            if fPath:
                #make Data class
                self.data = d.Data(filepath=fPath, lineSeparated=True)
                #open file
                f = open(fPath, "rb")
                #make counter to estimate voc size
                vocCounter = Counter()
                #make counter to keep track of number of lines to process
                line_counter = 0
                #estimate vocabulary size
                for line in f:
                    line_counter += 1
                    if line_counter <= num_lines:
                        clean = line.rstrip()
                        tokens = clean.split(" ")
                        for t in tokens:
                            vocCounter[t] += 1
                f.close()
                for i in range(self.num_epochs):
                        #if file text must be processed
                            #only relevant for first epoch
                        if not self.training_vectors and i == 0:
                            #counter to keep track of the number of lines to process
                            c = 0
                            #counter to keep track of lines to remove for testing
                            l = 0

                            #counter of words added to indices
                            cWord = 0
                            lWord = 0
                            #iterate through each line
                            for line in f:
                                c+=1
                                if c <= num_lines:
                                    #skip any line that is larger than max length
                                    if len(line.rstrip().split(" ")) < self.max_seq_length:
                                        #set counter for test items
                                        l+=1
                                        #annotate line
                                        self.data.annotateText(line)
                                        #tokenize most recent sentence
                                        cWord, lWord = self.data.getTokenized(self.data.rawSents[-1], cWord, lWord)
                                        print("current sentence in words", self.data.seqWords[-1])
                                        print("current sentence in lemmas", self.data.seqLemmas[-1])
                                        #convert most recent sentence to vector
                                        lemmaVector = pre.convertSentenceToVec(self.data.seqLemmas[-1], self.embeddingClass, self.w2vDimension)
                                        #pad
                                        lemmaVectorPadded = pre.padToConstant(lemmaVector, self.embeddingClass, self.max_seq_length)
                                        #keep every 10th example for testing
                                        if l % (num_lines/10) == 0:
                                            self.testing_vectors.append(lemmaVectorPadded)
                                        #otherwise run through LSTM as a training example
                                        else:
                                            #add to training collection
                                            self.training_vectors.append(lemmaVectorPadded)
                                            #loop through each word in sequence
                                                #train = current word (j)
                                                #label = next word (j + 1)
                                            self._training_step(lemmaVectorPadded)
                                            self.model.reset_states()
                                else:
                                    break

                        #if not processing text or not first epoch
                        else:
                            #shuffle
                            shuffle(self.training_vectors)

                            #iterate through all training instances
                            for sent in self.training_vectors:
                                #loop through each word in sequence
                                self._training_step(sent)
                                self.model.reset_states()
        #if learning embeddings
        else:
            print("not yet implemented")


    #tests the model on all sentences reserved for testing
    def test(self):
        allResults = []
        for test_item in self.testing_vectors:
            sentenceAccuracy = self._testing_step(test_item)
            allResults.append(sentenceAccuracy)
            self.model.reset_states()
        if len(allResults) == 0.0 or sum(allResults) == 0.0:
            averageAccuracy = 0
        else:
            averageAccuracy = e.accuracy(sum(allResults), len(allResults))
        print("final accuracy", str(averageAccuracy))


    #trains the model on one training sentence
    def _training_step(self, item):
        for j in range(self.max_seq_length-1):
            jPlus1 = self.embeddingClass.most_similar(positive=[item[j+1]], topn=1)[0][0]
            #bail on sentence when the next word is np.zeros
                #either because it's padding or it's not in the word2vec vocabulary
            if np.all(item[j+1] != np.zeros(self.w2vDimension)):
                print("time step", j)
                print("next word", jPlus1)
                #set gold label
                gold = np.zeros(self.vocSize)
                gold[self.data.vocLemmaToIDX[jPlus1]] = 1.0
                #take one training step
                self.model.train_on_batch(item[j].reshape((1,1,self.w2vDimension)), gold.reshape((1,self.vocSize)))
            else:
                break


    #tests the model on one testing sentence
    def _testing_step(self, item):
        sentence = []
        results = []
        for m in range(self.max_seq_length-1):
            #if current word is not padding
            if np.all(item[m] != np.zeros(self.w2vDimension)):
                #get distribution over vocabulary as predicted by model
                distribution = self.model.predict_on_batch(item[m].reshape(1,1,self.w2vDimension))
                #index of most likely word
                idx = np.argmax(distribution)
                #get predicted word
                closest_word = self.data.vocIDXtoLemma[idx]
                #actual word
                real_word = self.embeddingClass.most_similar(positive=[item[m+1]], topn=1)[0][0]
                #add predicted word to sentence accumulation list
                sentence.append(closest_word)
                print("predicted sentence so far", sentence)
                print("predicted next word", closest_word)
                print("actual next word", real_word)
                if real_word == closest_word:
                    results.append(1)
                else:
                    results.append(0)
            else:
                break
        sentenceAccuracy = e.accuracy(results.count(1), len(results))
        print("final sentence", sentence)
        print("sentence accuracy", str(sentenceAccuracy))
        return sentenceAccuracy


    def pickleData(self, vector):
        print("to be implemented")


    def unpickleData(self, vector):
        print("to be implemented")


    def saveModel(self):
        print("to be implemented")

    def loadModel(self):
        print("to be implemented")


####################################################################################
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
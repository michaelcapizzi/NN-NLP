from keras.models import *
from keras.layers import *
import numpy as np
np.random.seed(1982)  # to guarantee same randomizations each test
from random import shuffle
from collections import Counter
import pickle
import Utils.PreProcessing as pre
import Utils.Evaluation as eval
import Data as d
from itertools import izip
from keras.callbacks import EarlyStopping





#TODO how to use callbacks for earlystopping



#pickles a given vector to a given location
def pickleData(vector, location):
    f = open(location, "wb")
    pickle.dump(vector, f)
    f.close()

#################################################

#unpickles from a location
#vector = unpickleData(location)
def unpickleData(location):
    f = open(location, "rb")
    saved = pickle.load(f)
    f.close()
    return saved

#################################################

class LSTM_keras:
    """
    builds an LSTM using keras front end that can be used for language modeling or sentence segmentation.  See example: http://keras.io/examples/#sequence-classification-with-lstm
    :param purpose If `EOS`, used for determining end-of-sentence, if `LM` used for language modeling
    :param num_layers Number of `LSTM` layers
    :param embeddingLayerClass Embedding_keras class (see Embedding) if vectors are to be learned
    :param embeddingClass `gensim` class of `word2vec`
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
                 purpose="EOS",
                 num_layers=1,
                 embeddingLayerClass=None,
                 embeddingClass=None,
                 w2vDimension=None,
                 cSize=100,
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
        self.purpose = purpose
        self.num_lines = None
        self.embeddingLayerClass = embeddingLayerClass
        self.embeddingClass = embeddingClass
        self.num_epochs = num_epochs
        self.max_seq_length = max_seq_length
        self.training_vectors = training_vector
        self.testing_vectors = testing_vector
        self.data = None
        self.vocSize = None
        self.cSize = cSize
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model_json = None
        self.model = Sequential()
        if embeddingLayerClass:
            #get the dimensions from the embedding layer
            self.w2vDimension = embeddingLayerClass.w2vDimension
            #build the embedding layer
            self.model.add(embeddingLayerClass.layer)
            #note: masking layer not needed here as it is handled in embedding layer
        else:
            #get the dimensions from the arguments
            self.w2vDimension = w2vDimension
            #manually add a masking layer (to handle variable length of sequences)
                #http://keras.io/layers/core/#masking
            self.model.add(Masking(mask_value=np.zeros(self.w2vDimension), batch_input_shape=(1,1,self.w2vDimension)))
        #add the LSTM layer(s)
            #if there is an embedding layer
        if embeddingLayerClass:
            #for single layer
            if num_layers == 1:
                self.model.add(LSTM(
                        output_dim=self.cSize,
                        activation="tanh",
                        inner_activation="hard_sigmoid",
                        W_regularizer=self.W_regularizer,
                        U_regularizer=self.U_regularizer,
                        b_regularizer=self.b_regularizer,
                        dropout_W=self.dropout_W,
                        dropout_U=self.dropout_U,
                        return_sequences=False,           #True when using more than one layer
                        stateful=True,
                        # batch_input_shape=(1,1,self.w2vDimension)
                ))
            else:
                #for multilayer
                for i in range(num_layers):
                    #for first layer
                    if i == 0:
                        self.model.add(LSTM(
                                output_dim=self.cSize,
                                activation="tanh",
                                inner_activation="hard_sigmoid",
                                W_regularizer=self.W_regularizer,
                                U_regularizer=self.U_regularizer,
                                b_regularizer=self.b_regularizer,
                                dropout_W=self.dropout_W,
                                dropout_U=self.dropout_U,
                                return_sequences=True,           #True when using more than one layer
                                stateful=True,
                                # batch_input_shape=(1,1,self.w2vDimension)
                        ))
                    #for last layer
                    elif i == num_layers - 1:
                        self.model.add(LSTM(
                                output_dim=self.cSize,
                                activation="tanh",
                                inner_activation="hard_sigmoid",
                                W_regularizer=self.W_regularizer,
                                U_regularizer=self.U_regularizer,
                                b_regularizer=self.b_regularizer,
                                dropout_W=self.dropout_W,
                                dropout_U=self.dropout_U,
                                return_sequences=False,           #True when using more than one layer
                                stateful=True
                        ))
                    #for middle layer(s)
                    else:
                        self.model.add(LSTM(
                                output_dim=self.cSize,
                                activation="tanh",
                                inner_activation="hard_sigmoid",
                                W_regularizer=self.W_regularizer,
                                U_regularizer=self.U_regularizer,
                                b_regularizer=self.b_regularizer,
                                dropout_W=self.dropout_W,
                                dropout_U=self.dropout_U,
                                return_sequences=True,           #True when using more than one layer
                                stateful=True
                        ))
        #otherwise
        else:
            #add the LSTM layer(s)
            #for single layer
            if num_layers == 1:
                self.model.add(LSTM(
                    output_dim=self.cSize,
                    activation="tanh",
                    inner_activation="hard_sigmoid",
                    W_regularizer=self.W_regularizer,
                    U_regularizer=self.U_regularizer,
                    b_regularizer=self.b_regularizer,
                    dropout_W=self.dropout_W,
                    dropout_U=self.dropout_U,
                    return_sequences=False,           #True when using more than one layer
                    stateful=True,
                    batch_input_shape=(1,1,self.w2vDimension)
                ))
            else:
                #for multilayer
                for i in range(num_layers):
                    #for first layer
                    if i == 0:
                        self.model.add(LSTM(
                            output_dim=self.cSize,
                            activation="tanh",
                            inner_activation="hard_sigmoid",
                            W_regularizer=self.W_regularizer,
                            U_regularizer=self.U_regularizer,
                            b_regularizer=self.b_regularizer,
                            dropout_W=self.dropout_W,
                            dropout_U=self.dropout_U,
                            return_sequences=True,           #True when using more than one layer
                            stateful=True,
                            batch_input_shape=(1,1,self.w2vDimension)
                        ))
                    #for last layer
                    elif i == num_layers - 1:
                        self.model.add(LSTM(
                            output_dim=self.cSize,
                            activation="tanh",
                            inner_activation="hard_sigmoid",
                            W_regularizer=self.W_regularizer,
                            U_regularizer=self.U_regularizer,
                            b_regularizer=self.b_regularizer,
                            dropout_W=self.dropout_W,
                            dropout_U=self.dropout_U,
                            return_sequences=False,           #True when using more than one layer
                            stateful=True
                        ))
                    #for middle layer(s)
                    else:
                        self.model.add(LSTM(
                            output_dim=self.cSize,
                            activation="tanh",
                            inner_activation="hard_sigmoid",
                            W_regularizer=self.W_regularizer,
                            U_regularizer=self.U_regularizer,
                            b_regularizer=self.b_regularizer,
                            dropout_W=self.dropout_W,
                            dropout_U=self.dropout_U,
                            return_sequences=True,           #True when using more than one layer
                            stateful=True
                        ))

#################################################

    #must be done first!
        #num_lines = number of lines to process in file
            #if num_lines == 0: process entire file
    def prepareData(self, fPath, num_lines):
        if num_lines != 0:
            self.num_lines = num_lines
        #if model to be used for language modeling *and* an embedding layer,
            # vocSize has already been calculated
        if self.purpose == "LM" and self.embeddingLayerClass:
            #create data class
            self.data = d.Data(filepath=fPath, lineSeparated=True)
            self.vocSize = self.embeddingLayerClass.vocSize
        #else if model to be used for language modeling *but* no embedding layer,
            # vocSize must be calculated now
        elif self.purpose == "LM":
            #create data class
            self.data = d.Data(filepath=fPath, lineSeparated=True)
            #estimate vocabulary size
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
                self.num_lines = line_counter
            f.close()
            #set vocabulary size
            self.vocSize = len(vocCounter.items())
        #if model to be used for EOS detection
        elif self.purpose == "EOS":
            #create data class
            self.data = d.Data(filepath=fPath, lineSeparated=True)
            #open file
            f = open(fPath, "rb")
            #if reading whole file, get number of lines
            if num_lines == 0:
                line_counter = 0
                for line in f:
                    line_counter += 1
                f.close()
                self.num_lines = line_counter


#################################################


    #builds LSTM model
    #must be done **after** prepareData()
    def buildModel(self):
        #determine output dimension
        if self.purpose == "LM":
            outputDim = self.vocSize
        elif self.purpose == "EOS":
            outputDim = 2
        #default to language modeling
        else:
            outputDim = self.vocSize

        #add remaining layers
        # self.model.add(Dropout)
        self.model.add(Dense(
                output_dim=outputDim,
                activation="softmax"
        ))

        #compile
        self.model.compile(
                loss=self.loss_function,
                optimizer=self.optimizer,
                metrics=["accuracy"]
        )

        #print model summary
        self.model.summary()


#################################################


        #file = file to use for training
        #num_lines = number of lines to use from training file
            #0 = all
        #training_vector = vector of training instances
    def train(self, fPath):
        if not self.embeddingLayerClass:
            if fPath:
                #open file
                f = open(fPath, "rb")
                #start the server
                # print("starting processors server")
                self.data.startServer()
                for i in range(self.num_epochs):
                    #if file text must be processed
                        #only relevant for first epoch
                    if not self.training_vectors and i == 0:
                        #initialize vectors to house training and testing instances
                        self.training_vectors = []
                        self.testing_vectors = []
                        #counter to keep track of the number of lines to process
                        c = 0
                        #counter to keep track of lines to remove for testing
                        l = 0

                        #counter of words added to indices
                        cWord = 0
                        lWord = 0
                        #iterate through each line
                        for line in f:
                            if c <= self.num_lines:
                                #skip any line that is larger than max length
                                if len(line.rstrip().split(" ")) < self.max_seq_length:

                                    #annotate line
                                    self.data.annotateText(line)
                                    #tokenize most recent sentence
                                    cWord, lWord = self.data.getTokenized(self.data.rawSents[-1], cWord, lWord)
                                    print("current sentence in words", self.data.seqWords[-1])
                                    print("current sentence in lemmas", self.data.seqLemmas[-1])
                                    #convert most recent sentence to vector
                                    lemmaVector = pre.convertSentenceToVec(self.data.seqLemmas[-1], self.embeddingClass, self.w2vDimension)
                                    #only keep sentences for training and testing instances that have all words in word2vec list
                                    if np.all(lemmaVector != np.zeros(self.w2vDimension)):
                                        #set counter for total number of lines
                                        c+=1
                                        #set counter for test items
                                        l+=1

                                        #pad
                                        lemmaVectorPadded = pre.padEmbeddingToConstant(lemmaVector, self.w2vDimension, self.max_seq_length)
                                        #keep every 10th example for testing
                                        if l % (self.num_lines/10) == 0:
                                            self.testing_vectors.append(lemmaVectorPadded)
                                        #otherwise run through LSTM as a training example
                                        else:
                                            #add to training collection
                                            self.training_vectors.append(lemmaVectorPadded)
                                            #loop through each word in sequence
                                                #train = current word (j)
                                                #label = next word (j + 1) or EOS label
                                            if self.purpose == "LM":
                                                self._training_step_lm_w2v(lemmaVectorPadded)
                                            elif self.purpose == "EOS":
                                                self._training_step_eos_w2v(lemmaVectorPadded)
                                            else:
                                                self._training_step_lm_w2v(lemmaVectorPadded)
                            else:
                                break
                        f.close()
                        print("stopping server")
                        self.data.stopServer()

                    #if not processing text or not first epoch
                    else:
                        #shuffle
                        shuffle(self.training_vectors)

                        #iterate through all training instances
                        for k in range(len(self.training_vectors)):
                            if k % 100 == 0:
                                print("epoch", str(i+1))
                                print("training instance %s of %s" %(str(k+1), str(len(self.training_vectors))))
                            sent = self.training_vectors[i]
                            #loop through each word in sequence
                            if self.purpose == "LM":
                                self._training_step_lm_w2v(sent)
                            elif self.purpose == "EOS":
                                self._training_step_eos_w2v(sent)
                            else:
                                self._training_step_lm_w2v(sent)
            #if no file given, use loaded vectors
            else:
                for i in range(self.num_epochs):
                    #shuffle
                    shuffle(self.training_vectors)

                    #iterate through all training instances
                    for sent in self.training_vectors:
                        #loop through each word in sequence
                        if self.purpose == "LM":
                            self._training_step_lm_w2v(sent)
                        elif self.purpose == "EOS":
                            self._training_step_eos_w2v(sent)
                        else:
                            self._training_step_lm_w2v(sent)
        #if learning embeddings
        else:
            if fPath:
                #open file
                f = open(fPath, "rb")
                #start the server
                print("starting processors server")
                self.data.startServer()
                for i in range(self.num_epochs):
                    #must process text (also when loading pickled vectors) in order to get indices for lemma lookup
                    if i == 0:
                        #initialize vectors to house training and testing instances
                        self.training_vectors = []
                        self.testing_vectors = []
                        #counter to keep track of the number of lines to process
                        c = 0
                        #counter to keep track of lines to remove for testing
                        l = 0

                        #counter of words added to indices
                        cWord = 0
                        lWord = 0
                        #iterate through each line
                        for line in f:
                            if c <= self.num_lines:
                                #skip any line that is larger than max length
                                if len(line.rstrip().split(" ")) < self.max_seq_length:

                                    #annotate line
                                    self.data.annotateText(line)
                                    #tokenize most recent sentence
                                    cWord, lWord = self.data.getTokenized(self.data.rawSents[-1], cWord, lWord)
                                    print("current sentence in words", self.data.seqWords[-1])
                                    print("current sentence in lemmas", self.data.seqLemmas[-1])
                                    #convert most recent sentence to one-hots
                                    oneHotsVector = pre.convertSentenceToOneHots(self.data.seqLemmas[-1], self.data.vocLemmaToIDX)
                                    #only keep sentences for training and testing instances that have all words in lookup
                                    if None not in oneHotsVector:
                                        #set counter for total number of lines
                                        c+=1
                                        #set counter for test items
                                        l+=1

                                        #pad
                                        oneHotsVectorPadded = pre.padOneHotToConstant(oneHotsVector, self.max_seq_length)
                                        #keep every 10th example for testing
                                        if l % (self.num_lines/10) == 0:
                                            self.testing_vectors.append(oneHotsVectorPadded)
                                        #otherwise run through LSTM as a training example
                                        else:
                                            #add to training collection
                                            self.training_vectors.append(oneHotsVectorPadded)
                                            #loop through each word in sequence
                                                #train = current word (j)
                                                #label = next word (j + 1) or EOS label
                                            if self.purpose == "LM":
                                                self._training_step_lm_embed(oneHotsVectorPadded)
                                            elif self.purpose == "EOS":
                                                self._training_step_eos_embed(oneHotsVectorPadded)
                                            else:
                                                self._training_step_lm_embed(oneHotsVectorPadded)
                            else:
                                break
                        f.close()
                        print("stopping server")
                        self.data.stopServer()

                    #if not processing text or not first epoch
                    else:
                        #shuffle
                        shuffle(self.training_vectors)

                        #iterate through all training instances
                        for k in range(len(self.training_vectors)):
                            if k % 100 == 0:
                                print("epoch", str(i+1))
                                print("training instance %s of %s" %(str(k+1), str(len(self.training_vectors))))
                            sent = self.training_vectors[i]
                            #loop through each word in sequence
                            if self.purpose == "LM":
                                self._training_step_lm_embed(sent)
                            elif self.purpose == "EOS":
                                self._training_step_eos_embed(sent)
                            else:
                                self._training_step_lm_embed(sent)
            #if no file given, use loaded vectors
            else:
                for i in range(self.num_epochs):
                    #shuffle
                    shuffle(self.training_vectors)

                    #iterate through all training instances
                    for sent in self.training_vectors:
                        #loop through each word in sequence
                        if self.purpose == "LM":
                            self._training_step_lm_embed(sent)
                        elif self.purpose == "EOS":
                            self._training_step_eos_embed(sent)
                        else:
                            self._training_step_lm_embed(sent)

#################################################


    #tests the model for language modeling on all sentences reserved for testing
    def test_lm_w2v(self):
        if self.purpose == "EOS":
            print("run test_eos to properly test the model.")
        else:
            allResults = []
            for test_item in self.testing_vectors:
                sentenceAccuracy = self._testing_step_lm_w2v(test_item)
                allResults.append(sentenceAccuracy)
                self.model.reset_states()
            if len(allResults) == 0.0 or sum(allResults) == 0.0:
                averageAccuracy = 0
            else:
                averageAccuracy = eval.accuracy(sum(allResults), len(allResults))
            print("final accuracy", str(averageAccuracy))


    #tests the model for language modeling on all sentences reserved for testing
    def test_lm_embed(self):
        print("not yet implemented")


#################################################


    #tests the model for EOS detection on all sentences reserved for testing
    def test_eos_w2v(self):
        if self.purpose != "EOS":
            print("run test_lm to properly test the model.")
        else:
            allResults = []
            for test_item in self.testing_vectors:
                results = self._testing_step_eos_w2v(test_item)
                for r in results:
                    allResults.append(r)
                #final results
                finalTP = allResults.count("tp")
                finalTN = allResults.count("tn")
                finalFP = allResults.count("fp")
                finalFN = allResults.count("fn")
                finalPrecision = eval.precision(finalTP, finalFP)
                finalRecall = eval.recall(finalTP, finalFN)
                finalF1 = eval.f1(finalPrecision, finalRecall)
                print("final precision", finalPrecision)
                print("final recall", finalRecall)
                print("final f1", finalF1)


    #tests the model for EOS detection on all sentences reserved for testing
    def test_eos_embed(self):
        if self.purpose != "EOS":
            print("run test_lm to properly test the model.")
        else:
            allResults = []
            for test_item in self.testing_vectors:
                results = self._testing_step_eos_embed(test_item)
                for r in results:
                    allResults.append(r)
                #final results
                finalTP = allResults.count("tp")
                finalTN = allResults.count("tn")
                finalFP = allResults.count("fp")
                finalFN = allResults.count("fn")
                finalPrecision = eval.precision(finalTP, finalFP)
                finalRecall = eval.recall(finalTP, finalFN)
                finalF1 = eval.f1(finalPrecision, finalRecall)
                print("final precision", finalPrecision)
                print("final recall", finalRecall)
                print("final f1", finalF1)


#################################################


    #trains the model on one training sentence
    def _training_step_lm_w2v(self, item):
        for j in range(self.max_seq_length-1):
            jPlus1 = self.embeddingClass.most_similar(positive=[item[j+1]], topn=1)[0][0]
            #bail on sentence when the next word is np.zeros
                #either because it's padding or it's not in the word2vec vocabulary
            if np.all(item[j+1] != np.zeros(self.w2vDimension)):
                # print("time step", j)
                # print("next word", jPlus1)
                #set gold label
                gold = np.zeros(self.vocSize)
                gold[self.data.vocLemmaToIDX[jPlus1]] = 1.0
                #take one training step
                self.model.train_on_batch(item[j].reshape((1,1,self.w2vDimension)), gold.reshape((1,self.vocSize)))
            else:
                break
        #reset model cell state
        self.model.reset_states()


    #trains the model on one training sentence
    def _training_step_lm_embed(self, item):
        print("not yet implemented")


#################################################


    #tests the model on one testing sentence
    def _testing_step_lm_w2v(self, item):
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
        sentenceAccuracy = eval.accuracy(results.count(1), len(results))
        print("final sentence", sentence)
        print("sentence accuracy", str(sentenceAccuracy))
        return sentenceAccuracy


    #tests the model on one testing sentence
    def _testing_step_lm_embed(self, item):
        print("not yet implemented")



#################################################


    def _training_step_eos_w2v(self, item):
        for j in range(self.max_seq_length-1):
            if j == self.max_seq_length - 1 or np.all(item[j+1] == np.zeros(self.w2vDimension)):
                gold = np.array([1,0])
                self.model.train_on_batch(item[j].reshape(1,1,self.w2vDimension), gold.reshape(1,2))
                #reset model cell states
                self.model.reset_states()
                #break out of the training
                break
            else:
                gold = np.array([0,1])
                self.model.train_on_batch(item[j].reshape(1,1,self.w2vDimension), gold.reshape(1,2))


    def _training_step_eos_embed(self, item):
        for j in range(self.max_seq_length-1):
            if j == self.max_seq_length - 1 or item[j+1] == 0:
                gold = np.array([1,0])
                # self.model.train_on_batch(item[j].reshape(1,1), gold.reshape(1,2))
                self.model.train_on_batch(np.array([item[j]]), gold.reshape(1,2))
                #reset model cell states
                self.model.reset_states()
                #break out of the training
                break
            else:
                gold = np.array([0,1])
                # self.model.train_on_batch(item[j].reshape(1,1), gold.reshape(1,2))
                self.model.train_on_batch(np.array([item[j]]), gold.reshape(1,2))


#################################################


    def _testing_step_eos_w2v(self, item):
        sentence = []
        results = []
        for m in range(self.max_seq_length-1):
            #get distribution of lables
            distribution = self.model.predict_on_batch(item[m].reshape((1,1,self.w2vDimension)))
            #get argmax of softmax
            label = np.argmax(distribution)
            #get actual
            if m == self.max_seq_length - 1 or np.all(item[m+1] == np.zeros(self.w2vDimension)):
                actual = np.argmax(np.array([1,0]))
            else:
                actual = np.argmax(np.array([0,1]))
            #get the word associated with the current vector
            word = self.embeddingClass.most_similar(positive=[item[m]], topn=1)[0][0]
            #add to sentence
            sentence.append(word)
            print("current sentence", sentence)
            print("predicted label", label)
            print("actual label", actual)
            #record results
            if actual == 0 and label == 0:
                results.append("tp")
                print("end of sentence")
                break
            elif actual == 0 and label == 1:
                results.append("fn")
                print("end of sentence")
                break
            elif actual == 1 and label == 0:
                results.append("fp")
            else:
                results.append("tn")
        #reset model cell states
        self.model.reset_states()
        #get sentence results
        tp = results.count("tp")
        tn = results.count("tn")
        fp = results.count("fp")
        fn = results.count("fn")
        precision = eval.precision(tp, fp)
        recall = eval.recall(tp, fn)
        f1 = eval.f1(precision, recall)
        print("sentence precision", precision)
        print("sentence recall", recall)
        print("sentence f1", f1)
        return results


    def _testing_step_eos_embed(self, item):
        sentence = []
        results = []
        for m in range(self.max_seq_length-1):
            #get distribution of labels
            distribution = self.model.predict_on_batch(np.array([item[m]]))
            #get argmax of softmax
            label = np.argmax(distribution)
            #get actual
            if m == self.max_seq_length - 1 or item[m + 1] == 0:
                actual = np.argmax(np.array([1,0]))
            else:
                actual = np.argmax(np.array([0,1]))
            #get the word associated with the current vector
            word = self.data.vocIDXtoLemma.get(item[m])
            #add to sentence
            sentence.append(word)
            print("current sentence", sentence)
            print("predicted label", label)
            print("actual label", actual)
            #record results
            if actual == 0 and label == 0:
                results.append("tp")
                print("end of sentence")
                break
            elif actual == 0 and label == 1:
                results.append("fn")
                print("end of sentence")
                break
            elif actual == 1 and label == 0:
                results.append("fp")
            else:
                results.append("tn")
        #reset model cell states
        self.model.reset_states()
        #get sentence results
        tp = results.count("tp")
        tn = results.count("tn")
        fp = results.count("fp")
        fn = results.count("fn")
        precision = eval.precision(tp, fp)
        recall = eval.recall(tp, fn)
        f1 = eval.f1(precision, recall)
        print("sentence precision", precision)
        print("sentence recall", recall)
        print("sentence f1", f1)
        return results

#################################################

    #TODO currently not working
        #TODO try to.yaml()?
    #save model to .json
    def saveModel(self, location):
        self.model_json = self.model.to_json()
        open(location, "w").write(self.model_json)


#################################################

    #TODO test more
    #save weightse to .h5
    def saveWeights(self, location):
        self.model.save_weights(location)


#################################################

    #TODO currently not working
    #load model from .json
    def loadModel(self, location):
        self.model = model_from_json(location)

#################################################

    #TODO test more
    #load weights from .h5
    def loadWeights(self, location):
        self.model.load_weights(location)



#########################################################################################

class FF_keras:

    def __init__(self,
                 hidden_layer_dims=[100],
                 activations=["relu"],
                 embeddingClass=None,
                 w2vDimension=None,
                 window_size=3,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 loss_regularizer=None,
                 bias=True,
                 hidden_dropouts=[0],
                 loss_function="binary_crossentropy",
                 optimizer="adagrad",
                 num_epochs=5,
                 ):
        self.hidden_layer_dims=hidden_layer_dims
        self.activations=activations
        self.embeddingClass=embeddingClass
        self.w2vDimension=w2vDimension
        self.window_size=window_size
        self.W_regularizer=W_regularizer
        self.b_regularizer=b_regularizer
        self.W_constraint=W_constraint
        self.b_constraint=b_constraint
        self.loss_regularizer=loss_regularizer
        self.bias=bias
        self.hidden_dropouts=hidden_dropouts
        self.loss_function=loss_function
        self.optimizer=optimizer
        self.num_epochs=num_epochs
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model = Sequential()
        self.data=None
        self.processor=None
        self.training_vectors=[]
        self.training_labels=[]
        self.testing_vectors=[]
        self.testing_labels=[]



#################################################


    def buildModel(self):
        #for each layer
            #i = number of nodes in hidden layer
            #j = activation of hidden layer
            #k = dropout for hidden layer
        #counter for layers
        c = 0
        for i,j,k in zip(self.hidden_layer_dims,self.activations, self.hidden_dropouts):
            c+=1
            #for first layer
            if c == 1:
                #add dense with input_shape
                self.model.add(Dense(
                    output_dim=i,
                    init="lecun_uniform",
                    batch_input_shape=(1,self.w2vDimension*2*self.window_size)
                ))
                #add dropout
                self.model.add(Dropout(
                    p=k
                ))
                #add activation
                self.model.add(Activation(
                        activation=j
                ))
            #for last layer
            # elif c == len(self.hidden_layer_dims):
            #     #add dense
            #     self.model.add(Dense(
            #         output_dim=i,
            #         init="lecun_uniform"
            #     ))
            #     #add dropout
            #     self.model.add(Dropout(
            #         p=k
            #     ))
            #     #add activation
            #     self.model.add(Activation(
            #         activation=j
            #     ))
            #for middle layers
            else:
                #add dense
                self.model.add(Dense(
                    output_dim=i,
                    init="lecun_uniform"
                ))
                #add dropout
                self.model.add(Dropout(
                        p=k
                ))
                #add activation
                self.model.add(Activation(
                    activation=j
                ))

        #add final softmax layer
        self.model.add(Dense(
            output_dim=2
        ))
        self.model.add(Activation(
                activation="softmax"
        ))

        #compile
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_function,
                metrics=["accuracy"]
        )

        #print model summary
        self.model.summary()



#################################################

    #load data
        #takes OPEN files as arguments
        #training_cutoff = how data points of training to use
    def loadData(self, tr_vec, tr_lab, te_vec, te_lab, training_cutoff=0):
        #determine if a limited number of lines is to be used
        if training_cutoff != 0:
            cutoff = training_cutoff
        else:
            cutoff = None

        #line counter
        c = 0
        for v,l in izip(tr_vec, tr_lab):
            c+=1
            if not cutoff or c <= cutoff:
                vector = np.fromstring(v, sep=",")
                label = np.fromstring(l, sep=",")
                if c % 30000 == 0:
                    print("importing training data point number %s" %str(c))
                self.training_vectors.append(vector)
                self.training_labels.append(label)
            else:
                break
        tr_vec.close()
        tr_lab.close()

        c = 0
        for v,l in izip(te_vec, te_lab):
            c+=1
            vector = np.fromstring(v, sep=",")
            label = np.fromstring(l, sep=",")
            if c % 5000 == 0:
                print("importing testing data point number %s" %str(c))
            self.testing_vectors.append(vector)
            self.testing_labels.append(label)
        te_vec.close()
        te_lab.close()


    #file = file to use for training
    #num_lines = number of lines to use from training file
        #0 = all
    #negative sampling rate - double representing percentage of negative examples to *ignore*
    def train(self, fPath, num_lines, neg_sample, lemmatize=True):
        #negative sample threshold
        neg_cutoff = float(1000 * neg_sample)
        for e in range(self.num_epochs):
            #ignored negative datapoints
            neg_ignored = 0
            #pos examples
            pos = 0
            #neg examples
            neg = 0
            #if file text must be processed
            #only relevant for first epoch
            if fPath and e == 0:
                #open file
                f = open(fPath, "rb")
                #start the server
                self.processor = pre.initializeProcessor()
                #starting processors server
                pre.startServer(self.processor)
                #initialize vector to house training and testing instances
                self.training_vectors = []
                #counter to keep track of the number of lines to process
                c = 0
                #iterate through each line
                for line in f:
                    if (c <= num_lines or num_lines == 0) and len(line.split(" ")) > 1 and "@" not in line and "#" not in line:
                        #set counter for total number of lines
                        c+=1
                        try:
                            tokensLabels = pre.convertLineForEOS(line, self.processor, lemmatize)
                            print(line.rstrip(), c)
                            #process line
                            #unpack tokens and labels
                            tokens, labels = zip(*tokensLabels)
                            #convert tokens to vector representation
                            tokensVector = pre.convertSentenceToVec(tokens, self.embeddingClass, self.w2vDimension)
                            tokensVectorLabels = zip(tokensVector, labels)
                            [self.training_vectors.append(t) for t in tokensVectorLabels]
                        except Exception as ex:
                            print("ERROR in annotating.  Skipping line.")
                f.close()
                print("stopping server")
                self.processor.stop_server()
            #if not preprocessing
            for i in range(len(self.training_vectors)):
                slice_ = self.getSlice(self.training_vectors, i)
                label = self.training_labels[i].reshape((1,2))
                if i % 1000 == 0 or i == 0:
                    print("epoch", str(e + 1))
                    print("training instance %s of %s" %(str(i+1), str(len(self.training_vectors))))
                rand = np.random.randint(0,1000)
                #implement random negative resampling
                if np.argmax(label) == 0 or (np.argmax(label) == 1 and rand >= neg_cutoff):
                    self.model.train_on_batch(slice_.reshape(1,slice_.shape[0]), label)
                    #bookkeeping
                    if np.argmax(label) == 0:
                        pos+=1
                    else:
                        neg+=1
                else:
                    neg_ignored+=1
            print("number of ignored negative examples in epoch %s: %s" %(str(e + 1), str(neg_ignored)))
            print("positive examples in epoch %s: %s" %(str(e + 1), str(pos)))
            print("negative examples in epoch %s: %s" %(str(e + 1), str(neg)))


#################################################

    #test a trained model on test data file to be loaded
    def test(self, fPath, num_lines, lemmatize=True):
        results = []
        if fPath:
            self.testing_vectors = []
            #open file
            f = open(fPath, "rb")
            #start the server
            #start the server
            self.processor = pre.initializeProcessor()
            #starting processors server
            pre.startServer(self.processor)
            #initialize vector to house testing instances
            #counter to keep track of the number of lines to process
            c = 0
            #iterate through each line
            for line in f:
                if c <= num_lines or num_lines == 0:
                    #set counter for total number of lines
                    c+=1
                    #process line
                    tokensLabels = pre.convertLineForEOS(line, self.processor, lemmatize)
                    #unpack tokens and labels
                    tokens, labels = zip(*tokensLabels)
                    #convert tokens to vector representation
                    tokensVector = pre.convertSentenceToVec(tokens, self.embeddingClass, self.w2vDimension)
                    tokensVectorLabels = zip(tokensVector, labels)
                    [self.testing_vectors.append(t) for t in tokensVectorLabels]
            f.close()
            print("stopping server")
            self.processor.stop_server()
        for i in range(len(self.testing_vectors)):
            actual = self.testing_labels[i].reshape((1,2))
            slice_ = self.getSlice(self.testing_vectors, i)
            distribution = self.model.predict_on_batch(slice_.reshape(1,slice_.shape[0]))
            distArgMax = np.argmax(distribution)
            if distArgMax == 0:
                predicted = 1
            else:
                predicted = 0
            if np.argmax(actual) == 0 and predicted == 1:
                results.append("fp")
            elif np.argmax(actual) == 1 and predicted == 0:
                results.append("fn")
            elif np.argmax(actual) == 1 and predicted == 1:
                results.append("tp")
            if i % 5000 == 0 or i == 0:
                precision = eval.precision(results.count("tp"), results.count("fp"))
                recall = eval.recall(results.count("tp"), results.count("fn"))
                print("testing instance %s of %s" %(str(i+1), str(len(self.testing_vectors))))
                print("predicted distribution", distribution)
                print("predicted label", predicted)
                print("actual label", actual)
                print("current precision", precision)
                print("current recall", recall)
                print("current f1", eval.f1(precision, recall))
        finalPrecision = eval.precision(results.count("tp"), results.count("fp"))
        finalRecall = eval.recall(results.count("tp"), results.count("fn"))
        finalF1 = eval.f1(finalPrecision, finalRecall)
        print("final precision", finalPrecision)
        print("final recall", finalRecall)
        print("final f1", finalF1)





#################################################

    #generates one slice in the process of iterating through data with sliding window
    def getSlice(self, list, i):
        #j == start of slice, inclusive (i - 3)
        #k == end of slice, exclusive ==> is allowed to go beyond max indexing value (i + 3)
        j = i - self.window_size
        k = i + self.window_size
        #get slice (and pad if necessary)
        if j < 0:
            slice_ = list[0:k]
            #get size of left-padding
            diff = 0 - j
            #left pad
            for d in range(diff):
                slice_.insert(0, np.zeros(self.w2vDimension))
        else:
            slice_ = pre.padEmbeddingToConstant(list[j:k], self.w2vDimension, self.window_size * 2)

        return np.concatenate(slice_)





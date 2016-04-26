from gensim import models as g
import Models.Model as m
import sys

#sys.argv[1] = file to process for data
#sys.argv[2] = number of lines to take from file
#sys.argv[3] = word2vec file
#sys.argv[4] = max length of sentence to consider (currently using 30); needed to handle fixed size of input matrix
#sys.argv[5] = c_size
#sys.argv[6] = # of epochs
#sys.argv[7] = loss function
#sys.argv[8] = optimizer
#sys.argv[9] = pickle file for training data
#sys.argv[10] = pickle file for testing data

print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

model = m.LSTM_keras(embeddingClass=w2v, w2vDimension=int(len(w2v["the"])), max_seq_length=int(sys.argv[4]), cSize=int(sys.argv[5]), num_epochs=int(sys.argv[6]), loss_function=sys.argv[7], optimizer=sys.argv[8])

print("preparing data file")
model.prepareData(sys.argv[1], int(sys.argv[2]))

model.buildModel()

print("training")
model.train(sys.argv[1])

print("testing")
model.test_eos()

print("pickling training data")
model.pickleData(model.training_vectors, sys.argv[9])

print("pickling testing data")
model.pickleData(model.testing_vectors, sys.argv[10])

print("hyperparameters")
print("number of lines", sys.argv[2])
print("word2vec", sys.argv[3])
print("max sentence length", sys.argv[4])
print("c_size", sys.argv[5])
print("number of epochs", sys.argv[6])
print("loss function", sys.argv[7])
print("optimizer", sys.argv[8])

# print("saving weights")
# model.saveWeights("weights.h5")
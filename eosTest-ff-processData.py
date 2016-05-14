from gensim import models as g
import Models.Model as m
import Models.Embedding as e
import sys
import os

#script for running feed forward neural net over same training and testing data as used in original segmentor while also pickling data for later use

#sys.argv[1] = file to process for training data
#sys.argv[2] = file to process for testing data
#sys.argv[3] = number of lines to take from file
#sys.argv[4] = word2vec file (w2v_Gigaword.txt.gz, w2v_Goldberg.txt.gz)
#sys.argv[5] = lemmatize, boolean
#sys.argv[6] = hidden layer dimensions, *e.g.* '100 200 300'
#sys.argv[7] = hidden layer activations, *e.g.* 'tanh tanh relu'
#sys.argv[8] = hidden layer dropouts, *e.g.* '.5 .5 .5'
#sys.argv[9] = window size
#sys.argv[10] = # of epochs
#sys.argv[11] = loss function
#sys.argv[12] = optimizer
#sys.argv[13] = location to pickle training data
#sys.argv[14] = location to pickle testing data
#sys.argv[15] = OPTIONAL location of .h5 file to save weights


print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[4], binary=False)

num_lines = int(sys.argv[3])
w2vUsed = sys.argv[4]
lemmatize = bool(sys.argv[5])
hidden_layer_dims = [int(h) for h in sys.argv[6].split(" ")]
hidden_layer_activations = sys.argv[7].split(" ")
hidden_layer_dropouts = [float(d) for d in sys.argv[8].split(" ")]
window_size = int(sys.argv[9])
num_epochs = int(sys.argv[10])
loss_function = sys.argv[11]
optimizer = sys.argv[12]

model = m.FF_keras(hidden_layer_dims=hidden_layer_dims, activations=hidden_layer_activations, embeddingClass=w2v, w2vDimension=len(w2v["the"]), window_size=window_size,  hidden_dropouts=hidden_layer_dropouts, loss_function=loss_function, optimizer=optimizer, num_epochs=num_epochs)

model.buildModel()

print("training")
model.train(sys.argv[1], num_lines, lemmatize)

print("pickling training data")
m.pickleData(model.training_vectors, sys.argv[13])

print("testing")
model.test(sys.argv[2], num_lines, lemmatize)

print("pickling testing data")
m.pickleData(model.testing_vectors, sys.argv[14])

#save weights?
if len(sys.argv) == 16:
    print("saving weights")
    #TODO implement in FF_keras
    # model.saveWeights(sys.argv[15])
else:
    print("ending without saving weights")


print("hyperparameters")
print("number of lines", num_lines)
print("word2vec", w2vUsed)
print("lemmatize", str(lemmatize))
print("number of layers", len(hidden_layer_dims))
print("hidden layer dims", hidden_layer_dims)
print("hidden layer activations", hidden_layer_activations)
print("hidden layer dropouts", hidden_layer_dropouts)
print("window_size", window_size)
print("number of epochs", num_epochs)
print("loss function", loss_function)
print("optimizer", optimizer)


from gensim import models as g
import Models.Model as m
import sys
import pickle

#script for running feed forward neural net over same training and testing data as used in original segmentor

#loads data from .csv's

#sys.argv[1] = training vectors
#sys.argv[2] = training labels
#sys.argv[3] = testing vectors
#sys.argv[4] = testing labels
#sys.argv[5] = w2v used = "Gigaword" or "Goldberg"
#sys.argv[5] = hidden layer dimensions, *e.g.* '100 200 300'
#sys.argv[6] = hidden layer activations, *e.g.* 'tanh tanh relu'
#sys.argv[7] = hidden layer dropouts, *e.g.* '.5 .5 .5'
#sys.argv[8] = window size
#sys.argv[9] = # of epochs
#sys.argv[10] = loss function
#sys.argv[11] = optimizer
#sys.argv[12] = OPTIONAL location of .h5 file to save weights

training_vectors = open(sys.argv[1], "rb")
training_labels = open(sys.argv[2], "rb")
testing_vectors = open(sys.argv[3], "rb")
testing_labels = open(sys.argv[4], "rb")
if sys.argv[5].startswith("Giga"):
    w2v_size = 200
elif sys.argv[6].startswith("Gold"):
    w2v_size = 300
hidden_layer_dims = [int(h) for h in sys.argv[6].split(" ")]
hidden_layer_activations = sys.argv[7].split(" ")
hidden_layer_dropouts = [float(d) for d in sys.argv[8].split(" ")]
window_size = int(sys.argv[9])
num_epochs = int(sys.argv[10])
loss_function = sys.argv[11]
optimizer = sys.argv[12]
if len(sys.argv) == 14:
    weights_location = sys.argv[13]
else:
    weights_location = None

#build model
model = m.FF_keras(hidden_layer_dims=hidden_layer_dims, activations=hidden_layer_activations, embeddingClass=None, w2vDimension=w2v_size, window_size=window_size, hidden_dropouts=hidden_layer_dropouts, loss_function=loss_function, optimizer=optimizer, num_epochs=num_epochs)

model.buildModel()

print("loading data")
model.loadData(training_vectors, training_labels, testing_vectors, testing_labels)

print("training")
model.train(None, 0)

print("testing")
model.test(None, 0)

#save weights?
if weights_location:
    print("saving weights")
    #TODO implement in FF_keras
    # model.saveWeights(sys.argv[13])
else:
    print("ending without saving weights")


print("hyperparameters")
print("number of layers", len(hidden_layer_dims))
print("hidden layer dims", hidden_layer_dims)
print("hidden layer activations", hidden_layer_activations)
print("hidden layer dropouts", hidden_layer_dropouts)
print("window_size", window_size)
print("number of epochs", num_epochs)
print("loss function", loss_function)
print("optimizer", optimizer)


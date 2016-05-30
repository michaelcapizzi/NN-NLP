from gensim import models as g
import Models.Model as m
import sys
import pickle

#script for running feed forward neural net over same training and testing data as used in original segmentor

#sys.argv[1] = training data to load
#sys.argv[2] = testing data to load
#sys.argv[3] = word2vec file (w2v_Gigaword.txt.gz, w2v_Goldberg.txt.gz)
#sys.argv[4] = lemmatize, boolean
#sys.argv[5] = hidden layer dimensions, *e.g.* '100 200 300'
#sys.argv[6] = hidden layer activations, *e.g.* 'tanh tanh relu'
#sys.argv[7] = hidden layer dropouts, *e.g.* '.5 .5 .5'
#sys.argv[8] = window size
#sys.argv[9] = # of epochs
#sys.argv[11] = loss function
#sys.argv[12] = optimizer
#sys.argv[13] = OPTIONAL location of .h5 file to save weights


print("loading embeddings")
w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

training_vector_path_1 = open("training_instances/01-" + sys.argv[1], "rb")
training_vector_path_2 = open("training_instances/02-" + sys.argv[1], "rb")
testing_vector_path = open("training_instances/" + sys.argv[2])
w2vUsed = sys.argv[3]
lemmatize = bool(sys.argv[4])
hidden_layer_dims = [int(h) for h in sys.argv[5].split(" ")]
hidden_layer_activations = sys.argv[6].split(" ")
hidden_layer_dropouts = [float(d) for d in sys.argv[7].split(" ")]
window_size = int(sys.argv[8])
num_epochs = int(sys.argv[9])
loss_function = sys.argv[10]
optimizer = sys.argv[11]

model = m.FF_keras(hidden_layer_dims=hidden_layer_dims, activations=hidden_layer_activations, embeddingClass=w2v, w2vDimension=len(w2v["the"]), window_size=window_size, hidden_dropouts=hidden_layer_dropouts, loss_function=loss_function, optimizer=optimizer, num_epochs=num_epochs)

model.buildModel()

print("loading training data")
train_1 = pickle.load(training_vector_path_1)
train_2 = pickle.load(training_vector_path_2)
model.training_vectors = train_1 + train_2


# model.training_vectors = m.unpickleData("training_instances/coca_2500-wsj-swbd.pickle")

print("loading testing data")
model.testing_vectors = pickle.load(testing_vector_path)

# model.testing_vectors = m.unpickleData("training_instances/pitchTesting.pickle")


print("training")
model.train(None, None, lemmatize=lemmatize)

print("testing")
model.test(None, None, lemmatize=lemmatize)

#save weights?
if len(sys.argv) == 14:
    print("saving weights")
    #TODO implement in FF_keras
    # model.saveWeights(sys.argv[13])
else:
    print("ending without saving weights")


print("hyperparameters")
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


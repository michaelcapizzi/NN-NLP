from gensim import models as g
import Models.Model as m
import Models.Embedding as e
import sys

#script for when pre-processing of data is required (will pickle data in the process)

#sys.argv[1] = file to process for data
#sys.argv[2] = number of lines to take from file
#sys.argv[3] = word2vec file (w2v_Gigaword.txt.gz, w2v_Goldberg.txt.gz, embed_random,
#sys.argv[4] = max length of sentence to consider (currently using 30); needed to handle fixed size of input matrix
#sys.argv[5] = number of layers
#sys.argv[6] = c_size
#sys.argv[7] = # of epochs
#sys.argv[8] = loss function
#sys.argv[9] = optimizer
#sys.argv[10] = pickle file for training data
#sys.argv[11] = pickle file for testing data
#sys.argv[12] = OPTIONAL location of .h5 file to save weights


if "embed" not in sys.argv[3]:
    print("loading embeddings")
    w2v = g.Word2Vec.load_word2vec_format(sys.argv[3], binary=False)

    model = m.LSTM_keras(num_layers=int(sys.argv[5]), embeddingClass=w2v, w2vDimension=int(len(w2v["the"])), max_seq_length=int(sys.argv[4]), cSize=int(sys.argv[6]), num_epochs=int(sys.argv[7]), loss_function=sys.argv[8], optimizer=sys.argv[9])

else:
    embedding = e.Embedding_keras()

    print("calculating vocab size")
    embedding.getVocabSize(sys.argv[1], int(sys.argv[2]))
    print("vocab size", embedding.vocSize)

    print("building embedding layer")
    embedding.build()

    model = m.LSTM_keras(num_layers=int(sys.argv[5]), embeddingLayerClass=embedding, max_seq_length=int(sys.argv[4]), cSize=int(sys.argv[6]), num_epochs=int(sys.argv[7]), loss_function=sys.argv[8], optimizer=sys.argv[9])


print("preparing data file")
model.prepareData(sys.argv[1], int(sys.argv[2]))

model.buildModel()

print("training")
model.train(sys.argv[1])

print("testing")
if "embed" not in sys.argv[3]:
    model.test_eos_w2v()
else:
    model.test_eos_embed()

print("pickling training data")
m.pickleData(model.training_vectors, sys.argv[10])

print("pickling testing data")
m.pickleData(model.testing_vectors, sys.argv[11])

if len(sys.argv) == 13:
    print("saving weights")
    model.saveWeights(sys.argv[12])
else:
    print("ending without saving weights")

print("hyperparameters")
print("number of lines", sys.argv[2])
print("word2vec", sys.argv[3])
print("max sentence length", sys.argv[4])
print("number of layers", sys.argv[5])
print("c_size", sys.argv[6])
print("number of epochs", sys.argv[7])
print("loss function", sys.argv[8])
print("optimizer", sys.argv[9])


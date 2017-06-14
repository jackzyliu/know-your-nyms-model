import sys
import pickle
import random
from lstm_common import *

sys.argv.insert(1, '--cnn-mem')
sys.argv.insert(2, '512')
sys.argv.insert(3, '--cnn-seed')
sys.argv.insert(4, '3016748844')

from evaluation_common import *
from paths_lstm_classifier import load_model
from utils import *


# TODO: should allow parameter config file
EMBEDDINGS_DIM = 50
LSTM_HIDDEN_DIM = 60
EMPTY_PATH = ((0, 0, 0, 0),)


# word_pair: (x, y)
# classifier: model
# paths: {vectorized path: c}
# word_vectors: (x_vector, y_vector)
def one_prediction(word_pair, classifier, paths, word_vectors):
  
    # load model specs
    lemma_lookup = classifier.model['lemma_lookup']
    pos_lookup = classifier.model['pos_lookup']
    dep_lookup = classifier.model['dep_lookup']
    dir_lookup = classifier.model['dir_lookup']

    W1 = parameter(classifier.model['W1'])
    b1 = parameter(classifier.model['b1'])
    W2 = None
    b2 = None

    if classifier.num_hidden_layers == 1:
        W2 = parameter(classifier.model['W2'])
        b2 = parameter(classifier.model['b2'])
   
    paths_copy = paths
    if len(paths_copy) == 0:
        paths_copy[EMPTY_PATH] = 1

    # compute the averaged path
    num_paths = reduce(lambda x, y: x + y, paths_copy.itervalues())
    path_embeddings = [get_path_embedding(classifier.builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, p).npvalue() * count for p, count in paths_copy.iteritems()]
    input_vec = esum(path_embeddings) * (1.0 / num_paths)

    # Concatenate x and y embeddings
    input_vec = concatenate([word_vector[0], input_vec, word_vector[1]])
    
    h = W1 * input_vec + b1
 
    if classifier.num_hidden_layers == 1:
        h = W2 * tanh(h) + b2

    output = softmax(h)
   
    return output

# path_dict: word pair --> list of {raw dependency paths --> count}
# embedding_dict: word --> glove embedding

def model_prediction(model_file_prefix, target_relation, word_pairs, path_dict, embedding_dict):

    # Load the pre-trained model file
    print 'Loading model...'
    classifier, lemma_index, pos_index, dep_index, dir_index = load_model(model_file_prefix)
    
    # Making prediction from each word pair
    print 'Making predictions...'
    predictions = []
    for x, y in word_pairs:
        # get list of {path: count}
        try:
            paths = path_dict[(x, y)] 
        except KeyError:
            paths = {}  # empty dictionary if not found
        # vectorize path 
        paths = {vectorize_path(path, lemma_index, pos_index, dep_index, dir_index): count for path, count in paths.iteritems()}
        xv = embedding_dict[x] if x in embedding_dict.keys() else np.zro(LEMMA_HIDDEN_DIM)
        yv = embedding_dict[y] if y in embedding_dict.keys() else np.zro(LEMMA_HIDDEN_DIM)
        predictions.append(one_prediction((x, y), classifer, paths, (xv, yv)))
        renew_cg()
    return [y == target_relation for y in predictions]





if __name__ == '__main__':
    pass

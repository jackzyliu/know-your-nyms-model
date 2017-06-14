import sys
import os
import pickle
import random
from lstm_common import *

sys.argv.insert(1, '--cnn-mem')
sys.argv.insert(2, '512')
sys.argv.insert(3, '--cnn-seed')
sys.argv.insert(4, '3016748844')

from evaluation_common import *
from paths_lstm_classifier import load_model


# TODO: should allow parameter config file
TEMP_VOCAB_FILE='../temp/temp_vocab'


# glove_file_path: file path of the 50D glove file
# TODO: should be replaced with a database
def load_word_vectors(glove_file_path):
    with codecs.open(glove_file_path, 'r', 'utf-8') as f_in:
        w, v = zip(*[line.strip().split(' ', 1) for line in f_in]) 
    wv = np.loadtxt(v)
    d = dict(zip(w, wv))
    return d


# dp_file_path: file path of the dp dictionary: word pair --> raw dependency path
def load_path_dict(dp_file_path):
    pass


# create a temporary vocabulary file for the parser (see below)
def create_temp_vocab_file(words, filename=TEMP_VOCAB_FILE):
    with open(filename, 'a') as f:
        for word in words:
           f.write("%s\n" % word) 

def delete_temp_vocab_file(filename=TEMP_VOCAB_FILE):
    os.remove(TEMP_VOCAB_FILE)

def extract_new_paths(parser_file, wiki_file, vocab_file, out_file):
    os.system('python ' + parser_file + ' ' + wiki_file + ' ' + vocab_file + ' ' + out_file)
    print 'Parsing Complete.'
        

if __name__ == '__main__':
    create_temp_vocab_file(['tree', 'leaf', 'wheel', 'car'])
    extract_new_paths('~/meronym/know-your-nyms/corpus/parse_wikipedia.py', '/nlp/users/zheyuan/wiki/enwiki-20170220-pages-meta-current1.xml-p000000010p000030303', TEMP_VOCAB_FILE, 'test_triplet')
    delete_temp_vocab_file()

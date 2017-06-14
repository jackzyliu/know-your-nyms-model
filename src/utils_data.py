import os
import sys
import time
import pickle
import random
import itertools
import numpy as np
import pandas as pd

from os.path import isfile, join
from collections import defaultdict

sys.path.append('../')
from CONFIG import *

# TODO: access to database to dump the latest dataset
# saved locally as dump.tsv
# processed dataset is stored in data_snapshots/yyymmdd/data.tsv
# format: base word \t input word \t relation
# simultaneously creates base_words.tsv and vocab.tsv
# input: 
# output: folder_path e.g. ~/meronym/know-your-nyms/data/snapshot/20170610
def download_dataset(train_cnt_threshold=2, predict_cnt_threshold=5):
    #date = time.strftime('%Y%m%d') 
    date = '20170610'
    folder_path = '{}/{}'.format(DATA_SNAPSHOT, date)     
 
    df = pd.DataFrame.from_csv(join(folder_path, 'dump.tsv'), sep='\t', header=0, index_col=None)
    df_train = df[df['CNT'] > train_cnt_threshold] 
    df_train = df_train[['BASE_WORD', 'INPUT_WORD', 'TYPE']]
    df_train.to_csv(join(folder_path, 'data.tsv'), sep='\t', header=False, index=False)
    
    # base_words.tsv & vocab.tsv 
    df_predict = df[df['CNT'] > predict_cnt_threshold]
    base_words = set(df_predict['BASE_WORD']) 
    vocab = base_words.copy()
    vocab.update(df_predict['INPUT_WORD'])
    
    with open(join(folder_path, 'base_word.tsv'), 'w') as f_b:
        for a in base_words:
            f_b.write(a + '\n')
    with open(join(folder_path, 'vocab.tsv'), 'w') as f_v:
        for b in vocab:
            f_v.write(b + '\n')
    return folder_path



# filter a dataset according to LexNET - remove word pairs with words that are not recognized by LexNET
# input: input dataset, vocab reference file, output dataset
# output: none
def filter_dataset_by_vocab(input_data_file, reference_file, output_data_file):
    entities_set = set()
    with open(reference_file, 'r') as f:
        for line in f.readlines():
            entities_set.add(line.rstrip('\n'))

    with open(input_data_file, 'r') as f_in, open(output_data_file, 'w') as f_out:
        for line in f_in.readlines():
            old_l = line.rstrip('\n').split('\t')
            if old_l[0] in entities_set and old_l[1] in entities_set:
                f_out.write(line)
            

# append tsv files 
# input: list of input data tsv files, output data file
# output: none
def append_datasets(input_data_files, output_data_file):
    with open(output_data_file, 'w') as f_out:
        for input_data_file in input_data_files:
            if not isfile(input_data_file) or not input_data_file.endswith('.tsv'):
                continue
            with open(input_data_file, 'r') as f_in:
                f_out.writelines(f_in.readlines())
 

# split a dataset into train, test, val according to a given proportion
# input: input dataset, proportions, output directory  
# output: none
def split_dataset(input_data_file, ratio_tuple, output_data_dir):
    tsv_files = ['train.tsv', 'val.tsv', 'test.tsv']
    # prepare ratios
    assert len(ratio_tuple) < 4 and len(ratio_tuple) >= 1
    ratios = [0]
    ratios.extend(list(ratio_tuple))
    ratios = np.array(ratios) / np.sum(ratios) # normalize
    ratios = np.cumsum(ratios) # cumulative sumi e.g. [0, 0.5, 1]
    # prepare splitting indices
    total_size = sum(1 for line in open(input_data_file, 'r'))
    indices = np.random.permutation(total_size)
    splits = (total_size * ratios).astype(np.int)
    splits = [indices[splits[i]:splits[i+1]] for i in range(len(tsv_files))]
    d = dict(zip(tsv_files, splits))
    # output
    for tsv_file in tsv_files:
        print tsv_file
        with open(join(output_data_dir, tsv_file), 'w') as f_out, open(input_data_file, 'r') as f_in:
            target_indices = d[tsv_file]    
            line_num = 0
            for line in f_in.readlines():
                if line_num in target_indices:
                    f_out.write(line)
                line_num += 1


# prints out summarization the dataset 
# input: dataset file
# output: none
def summarize_dataset(input_data_file):
    relations_count = defaultdict(int)
    with open(input_data_file, 'r') as f:
        for line in f.readlines():
            old_l = line.rstrip('\n').split('\t')
            relations_count[old_l[-1]] += 1
    counts = pd.DataFrame.from_dict(relations_count, orient='index') 
    counts.rename(columns={0: 'counts'}, inplace=True)
    counts['perc'] = counts['counts']/counts['counts'].sum()


# create a relations.txt file from a dataset file
# input: dataset file, output dir
# output: 
def create_relations_file(input_data_file, output_data_dir):
    filename = 'relations.txt'
    relations = set()
    with open(input_data_file, 'r') as f_in, open(join(output_data_dir, filename), 'w') as f_out: 
        for line in f_in.readlines():
            l = line.rstrip('\n').split('\t')
            if l[-1] not in relations:
                relations.add(l[-1])
                f_out.write(l[-1] + '\n')
     

# create a predict file by combining the base words and the vocab
def create_predict_dataset(input_data_dir, output_data_dir):
    base_filename = 'base_word.tsv'
    vocab_filename = 'vocab.tsv'
    out_filename = 'predict.tsv'
    with open(join(input_data_dir, base_filename), 'r') as f_base, \
         open(join(input_data_dir, vocab_filename), 'r') as f_vocab: 
        content = list(itertools.product(f_base.readlines(), f_vocab.readlines()))
    with open(join(output_data_dir, out_filename), 'w') as f_out:
        for x, y in content:
            x = x.strip().lower()
            y = y.strip().lower()
            if x == y:
                continue
            f_out.write('{}\t{}\n'.format(x, y))


if __name__ == '__main__':
    p = download_dataset(predict_cnt_threshold=10) 
    append_datasets([p + '/data.tsv', DATA_REFERENCE + '/Random/data.tsv'], DATA_MODEL + '/data.tsv')
    summarize_dataset(DATA_MODEL + '/data.tsv')
    filter_dataset_by_vocab(DATA_MODEL + '/data.tsv', WIKI_ENTITIES, DATA_MODEL+ '/filtered_data.tsv') 
    split_dataset(DATA_MODEL + '/filtered_data.tsv', [0.75, 0.2, 0.05], DATA_MODEL)
    create_relations_file(DATA_MODEL + '/train.tsv', DATA_MODEL)
    create_predict_dataset('../data/snapshot/20170610', DATA_MODEL) 

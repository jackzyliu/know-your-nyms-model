import os
import sys
import time

from os.path import isfile, join
from datetime import timedelta

sys.path.append('../')
from CONFIG import *

# this function encloses a call to train_integrated.py with already supplied
# arguments, can be used to train the model in the background
def train():
    date = time.strftime("%Y%m%d") 
    log_file = join(LOG, date) 
    os.system('echo Model Training... >> {}'.format(log_file))
    command = 'python {}/train_integrated.py {} {} {} {} {}'.format(LEXNET, CORPUS, DATA_MODEL, MODEL_OUTPUT, GLOVE, str(NUM_HIDDEN_LAYERS))
    start_time = time.time()
    os.system('{} 1>>{} 2>>{}'.format(command, log_file, log_file))
    elapsed = time.time() - start_time
    duration = str(timedelta(seconds=elapsed))
    os.system('echo Total Time Used: {} >> {}'.format(duration, log_file))


if __name__ == '__main__':
    train()

# folder paths
PROJECT='/home1/z/zheyuan/meronym/know-your-nyms' # CHANGE
DATA_SNAPSHOT=PROJECT + '/data/snapshot'
DATA_REFERENCE=PROJECT + '/data/reference'
DATA_MODEL=PROJECT + '/data/model'
LEXNET=PROJECT + '/lexnet'
LOG=PROJECT + '/log'

# file paths
WIKI_ENTITIES=DATA_REFERENCE + '/wiki_paths_and_entities/wikiEntities.txt'
GLOVE='/nlp/users/zheyuan/glove/glove.6B.50d.txt' # CHANGE
DATA_PREDICT=DATA_MODEL + '/predict.tsv'

# prefix paths
CORPUS='/nlp/users/zheyuan/lexnet_wiki/wiki' # CHANGE
MODEL_OUTPUT='/nlp/users/zheyuan/know-your-nyms/model_output/test' # CHANGE

# model parameters
NUM_HIDDEN_LAYERS=0
EMBEDDINGS_DIM=50
LSTM_HIDDEN_DIM=60
EMPTY_PATH=((0, 0, 0, 0),)

# classifier parameters
NUM_LAYERS = 2
LEMMA_DIM = EMBEDDINGS_DIM
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1
LOSS_EPSILON = 0.01



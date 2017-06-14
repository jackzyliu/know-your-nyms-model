Description
This package is built for the purpose of interacting with know-your-nyms.com by periodically retrieving data from the website to train a centralized semantic relations classification model, and supplying classification results back to know-your-nyms.com. 


Package Map

data:
-snapshots: contains dumps of database data
-reference: contains different data sources
-model: contains data files used for model training, tuning, and testing

lexnet:
-evaluation_common.py
-lstm_common.py
-knowledge_resource.py
-paths_lstm_classifier.py

src:
-utils_data.py:
--download_dataset
--filter_dataset_by_vocab
--append_datasets
--split_dataset
--summarize_dataset
--create_relations_file 
--create_predict_dataset
-model_train.py
--train
-model_predict.py
--predict

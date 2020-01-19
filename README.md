This repository is handwriting recognition for english.

## Pre-requisites

- python 3.6
- Tensorflow
- keras 
- OpenCV

## dataset prepare
1. prepare fix height train samples for letters, words or sentences.
2. run data_proceess/data_letter_generate.py create single letter or multiple letters samples with 32*120.
3. run data_proceess/data_words_generate.py  create english words samples with 64*400.
4. run create_lmdb_dataset.sh can convert image files to lmdb format. 

## train model
python train.py 

You can choose a best model for your work.
1.single letter or multiple letters recognition:  simple_densenet.py ====> Densenet+CTC
2.words recognition:  densenet_blstm.py ====>Densenet+Bilstm+CTC
3.a new test for cnn: efficientnet_b0.py ====> efficientnet_b0+Bilstm+CTC


## test model

python test_model.py

You can choose one method for test.

flag=0 # test single or multiple imges
flag=1 # test file_lst and compute test precision
flag=2 # find hard samples

## test precision
-letters: Densenet+CTC  99.8%;
-words: Densenet+Bilstm+CTC  99.0%.
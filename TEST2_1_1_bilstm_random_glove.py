from Train2_1_1_bilstm_random_glove import *

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--initialization', type=str)
parser.add_argument('--model_file', type=str)
parser.add_argument('--test_data_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--glove_embeddings_file', type=str)
parser.add_argument('--vocabulary_input_file', type = str)
args=parser.parse_args()

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import io
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# import seqeval
# from seqeval.metrics import accuracy_score as seq_accuracy_score
# from seqeval.metrics import classification_report as seq_classification_report
# from seqeval.metrics import f1_score as seq_f1_score
import pickle as pickle


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  

def test_model(model, loader):
    y_predicted = []
    with torch.no_grad():
        for step, (X, Y, xlen) in enumerate(loader):
            ypred = model(X.long().to(device), xlen.to(device))#.permute(0, 2, 1)
            ypred = torch.argmax(ypred.to('cpu'), dim = 1)
            ypred = ypred.view(Y.shape[0], -1)
            y_predicted.append(ypred)

    y_predicted_list = []
    for i in range(len(y_predicted)):
        for j in range(y_predicted[i].shape[0]):
            sent_pred = []
            for x in range(y_predicted[i].shape[1]):
                sent_pred.append(id2tag[int(y_predicted[i][j, x])])
            y_predicted_list.append(sent_pred)
    return y_predicted_list
    #CONVERTING y_predicted and y_true lists into tag list
    # return y_predicted, y_true



#load model
# model = BiLSTM()
# model = torch.load(args.model_file,  map_location=torch.device('cpu'))
model = torch.load(args.model_file,map_location=torch.device(device))
model.eval()

[vocab, nertags] = pickle.load(open(args.vocabulary_input_file, "rb"))

#make id2tag
id2tag = {}
for tag in nertags.keys():
    if(tag == 'padtag'):
        id2tag[nertags[tag]] = 'O' # because we dont want the model to predict 'padtag' tags
    else:
        id2tag[nertags[tag]] = tag


# load test dataset
#Test DATASET
testdatapath = args.test_data_file
Xtest, Ytest, x_testlengths, _, _ = load_data(testdatapath, buildvocab_tags=False, vocab = vocab, nertags = nertags)

testdataset = TensorDataset(Xtest, Ytest, x_testlengths)
loader_test = DataLoader(testdataset, batch_size= 1, shuffle=False)

predictions = test_model(model, loader_test) #list of lists having predicted ner tags

#writing to a file and saving it
def writefile(testfilepath, outputfilepath, predictions):
    final_output = [] #list of lists which will finally be written to file
    with open(testfilepath) as f:
        lines = f.readlines()
        sentnum = -1 #to take care of the first blank line
        wordnum = 0
        for line in lines:
            if(line == '\n'):
                sentnum+=1
                wordnum = 0
                final_output.append(line)

            else:
                line_sep = line.split(sep = " ")
                x = line_sep[:-1]
                x.append(predictions[sentnum][wordnum]+'\n')
                wordnum+=1
                final_output.append(" ".join(x))
    #write the outputfilepath
    with open(outputfilepath, 'w+') as f:
        f.writelines(final_output)

writefile(testdatapath, args.output_file, predictions)
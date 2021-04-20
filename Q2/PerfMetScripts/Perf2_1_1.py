from Train2_1_1_bilstm_random_glove import *

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str)
parser.add_argument('--rootpath', type=str)
parser.add_argument('--Expname', type=str)
parser.add_argument('--train_data_file', type=str)
parser.add_argument('--val_data_file', type=str)
parser.add_argument('--test_data_file', type=str)
parser.add_argument('--vocabulary_input_file', type = str)
args=parser.parse_args()

import os
if not os.path.exists(args.rootpath):
    os.mkdir(args.rootpath)

if not os.path.exists(args.rootpath+args.Expname):
    os.mkdir(args.rootpath+args.Expname)


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
import seqeval
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
import pickle as pickle



if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  


model = torch.load(args.model_file,map_location=torch.device(device))
model.eval()

[vocab, nertags] = pickle.load(open(args.vocabulary_input_file, "rb"))


# def SavePlots(y1, y2, metric, rootpath, Expname):
#     try:
#         plt.clf()
#     except Exception as e:
#         pass
#     """y2 should be validation"""
#     epochs=np.arange(1,len(y1)+1,1)
#     plt.title(Expname + " " + metric + " plot")
#     plt.xlabel('Epochs')
#     plt.ylabel(metric)
#     plt.plot(epochs,y1,label='Training %s'%metric, linewidth = 2)
#     plt.plot(epochs,y2,label='Validation %s'%metric, linewidth = 2)
#     if(metric == "Loss"):
#         ep=np.argmin(y2)
#     elif(metric != "Loss"):
#         ep =np.argmax(y2)
#     plt.plot(ep+1,y2[ep],'r*',label='bestvalue@(%.i,%.2f)'%(ep+1,y2[ep]))
#     plt.grid()
#     plt.legend()
#     plt.savefig(args.rootpath+args.Expname+"/{}".format(metric), dpi=300)


# SavePlots(trainlosslist, validlosslist, "Loss", args.rootpath, args.Expname)
# SavePlots(trainacclist, valacclist, "Accuracy", args.rootpath, args.Expname)
# SavePlots(trainmicrof1list, valmicrof1list, "Micro F1", args.rootpath, args.Expname)
# SavePlots(trainmacrof1list, valmacrof1list, "Macro F1", args.rootpath, args.Expname)

#make id2tag
id2tag = {}
for tag in nertags.keys():
    if(tag == 'padtag'):
        id2tag[nertags[tag]] = 'O' # because we dont want the model to predict 'padtag' tags
    else:
        id2tag[nertags[tag]] = tag


def final_metrics(model, loader):
    y_predicted = []
    y_true = []
    with torch.no_grad():
        for step, (X, Y, xlen) in enumerate(loader):
            Y = pack_padded_sequence(Y, xlen, batch_first=True, enforce_sorted=False)
            Y, _ = pad_packed_sequence(Y, batch_first=True)
            ypred = model(X.long().to(device), xlen.to(device))#.permute(0, 2, 1)
            ypred = torch.argmax(ypred.to('cpu'), dim = 1)
            ypred = ypred.view(Y.shape[0], -1)
            y_predicted.append(ypred)
            y_true.append(Y)

    y_predicted_list = []
    y_true_list = []
    for i in range(len(y_predicted)):
        for j in range(y_predicted[i].shape[0]):
            sent_pred = []
            sent_true = []
            for x in range(y_predicted[i].shape[1]):
                sent_pred.append(id2tag[int(y_predicted[i][j, x])])
                sent_true.append(id2tag[int(y_true[i][j, x])])
            y_predicted_list.append(sent_pred)
            y_true_list.append(sent_true)
    return seq_f1_score(y_true_list, y_predicted_list), seq_accuracy_score(y_true_list, y_predicted_list), seq_classification_report(y_true_list, y_predicted_list, digits = 3)

# # calculate the final metrics usign seq eval
# # TRAINING DATA
# loader_train = DataLoader(traindataset, batch_size= 1, shuffle=False)
# train_f1_conll, train_acc_conll, train_classif_report = final_metrics(model, loader_train)

# # VALIDATION DATA
# loader_valid = DataLoader(devdataset, batch_size= 1, shuffle=False)
# valid_f1_conll, valid_acc_conll, valid_classif_report = final_metrics(model, loader_valid)

# print("PERFORMANCE ON Train DATA")
# print('MicroF1 = {}'.format(train_f1_conll))
# print('Accuracy = {}'.format(train_acc_conll))
# print('------------Classification Report-------------')
# print(train_classif_report)

# print("PERFORMANCE ON Validation DATA")
# print('MicroF1 = {}'.format(valid_f1_conll))
# print('Accuracy = {}'.format(valid_acc_conll))
# print('------------Classification Report-------------')
# print(valid_classif_report)

#Test DATASET
testdatapath = args.test_data_file
Xtest, Ytest, x_testlengths, _, _ = load_data(testdatapath, buildvocab_tags=False, vocab = vocab, nertags = nertags)

testdataset = TensorDataset(Xtest, Ytest, x_testlengths)
loader_test = DataLoader(testdataset, batch_size= 1, shuffle=False)

test_f1_conll, test_acc_conll, test_classif_report = final_metrics(model, loader_test)

print("PERFORMANCE ON Test DATA")
print('MicroF1 = {}'.format(test_f1_conll))
print('Accuracy = {}'.format(test_acc_conll))
print('------------Classification Report-------------')
print(test_classif_report)


"""SAVING DATA"""

# save performance metrics dictionaries
# save train loss, acc, micro, macro
# save val loss, acc, micro, macro
# save model
# import pickle
# #train
# pickle.dump(train_classif_report, open(args.rootpath+args.Expname+"/train_classif_report.dict.pickle", "wb" ))
# np.save(args.rootpath+args.Expname+"/train_losslist.npy", np.asarray(trainlosslist))
# np.save(args.rootpath+args.Expname+"/train_acclist.npy", np.asarray(trainacclist))
# np.save(args.rootpath+args.Expname+"/train_microf1list.npy", np.asarray(trainmicrof1list))
# np.save(args.rootpath+args.Expname+"/train_macrof1list.npy", np.asarray(trainmacrof1list))

# #valid
# pickle.dump(valid_classif_report, open(args.rootpath+args.Expname+"/valid_classif_report.dict.pickle", "wb" ))
# np.save(args.rootpath+args.Expname+"/val_losslist.npy", np.asarray(validlosslist))
# np.save(args.rootpath+args.Expname+"/val_acclist.npy", np.asarray(valacclist))
# np.save(args.rootpath+args.Expname+"/val_microf1list.npy", np.asarray(valmicrof1list))
# np.save(args.rootpath+args.Expname+"/val_macrof1list.npy", np.asarray(valmacrof1list))

# #test
# pickle.dump(test_classif_report, open(args.rootpath+args.Expname+"/test_classif_report.dict.pickle", "wb" ))